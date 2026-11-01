//! Fourier ring correlation of one image, as the Python package frc computes it with `one_frc`
//! and method 1: split every count n into k ~ binomial (n, 0.5) and n - k, Fourier transform both
//! halves, and correlate them over rings of equal distance from the origin. The FRC at ring r is
//! Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2), for r < N / 2.
//!
//! Every step is bound by memory bandwidth, so the steps are arranged to move the image as few
//! times as possible:
//!
//! - Both halves go through one complex FFT, of z = k + i (n - k): their spectra are
//!   F1(f) = (Z(f) + conj Z(-f)) / 2 and F2(f) = (Z(f) - conj Z(-f)) / 2i.
//! - The split writes z a few rows at a time and transforms those rows while they are in cache.
//! - The column FFTs copy 16 columns at a time into a buffer that stays in cache, so there is no
//!   transpose pass.
//! - F1 and F2 of a real image satisfy F(-f) = conj F(f), so the ring sums run over half the plane.
//!
//! Ring r holds the frequencies f with round(|f|) = r. That DIPlib's `RadialSum`, which frc uses,
//! bins the same way is not checked here.

use std::sync::Arc;

use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::ToUsizeError;
use crate::binom_variants::{CpuFeatures, LanesSplitter};

/// The sums over one ring of the spectra F1 and F2 of the two halves.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingSums {
    /// Re(sum F1 conj F2).
    pub correlation: f64,
    /// sum |F1|^2.
    pub power1: f64,
    /// sum |F2|^2.
    pub power2: f64,
    /// The number of frequencies in the ring.
    pub count: u64,
}

impl RingSums {
    pub fn frc(&self) -> f64 {
        self.correlation / (self.power1 * self.power2).sqrt()
    }
}

/// Counts that one task of the split samples: a chunk of `LanesSplitter`, so whole rows of them.
const SPLIT_TASK_COUNTS: usize = 16384;

/// Columns that the column FFTs copy into one buffer: 16 complex f32 are two cache lines of a row.
const COLUMN_BLOCK: usize = 16;

/// Complex f32s between the starts of two rows of the column buffer beyond the row itself: one
/// cache line, so that rows do not start a multiple of 4 KiB apart, where the 16 values that one
/// image row adds to them would all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// Rows ahead of the current tile that the column copies prefetch. Rows 32 KiB apart defeat the
/// hardware prefetchers, which follow streams within a 4 KiB page.
const PREFETCH_ROWS: usize = 64;

/// What `one_frc` needs for images of one size, made once like an FFT plan.
pub struct FrcPlan {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    /// For each frequency of the half plane fy > 0 or (fy = 0, fx > 0) with r = round(|f|) < N/2,
    /// its ring r; every other entry is `OUTSIDE_HALF_DISK`.
    half_plane_rings: Vec<u16>,
    ring_counts: Vec<u64>,
    /// The halves as z = k + i (n - k), then the spectrum Z.
    spectrum: Vec<Complex32>,
}

const OUTSIDE_HALF_DISK: u16 = u16::MAX;

/// The signed frequency of index i of a DFT of length n.
fn frequency(i: usize, n: usize) -> i64 {
    if i <= (n - 1) / 2 {
        i as i64
    } else {
        i as i64 - n as i64
    }
}

fn ring_of(fy: i64, fx: i64) -> usize {
    ((fy * fy + fx * fx) as f64).sqrt().round() as usize
}

/// `FrcPlan::spectrum` shared by the tasks of the column FFTs, which each touch their own
/// columns only.
#[derive(Clone, Copy)]
struct SpectrumColumns(*mut Complex32);
// SAFETY: the tasks that share it read and write disjoint columns.
unsafe impl Send for SpectrumColumns {}
unsafe impl Sync for SpectrumColumns {}

impl SpectrumColumns {
    fn ptr(self) -> *mut Complex32 {
        self.0
    }
}

/// Asks for the two cache lines of columns x0..x0 + 16 of rows y..y + 16 that exist.
#[inline(always)]
fn prefetch_column_tile(p: *mut Complex32, n: usize, x0: usize, y: usize) {
    #[cfg(target_arch = "x86_64")]
    for row in y..(y + COLUMN_BLOCK).min(n) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: a prefetch accesses no memory, and the address lies within the spectrum.
        unsafe {
            let line = p.add(row * n + x0) as *const i8;
            _mm_prefetch(line, _MM_HINT_T0);
            _mm_prefetch(line.add(64), _MM_HINT_T0);
        }
    }
}

impl FrcPlan {
    /// A plan for images of n x n counts.
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n / 2 < OUTSIDE_HALF_DISK as usize,
            "unsupported image size {n}"
        );
        let mut half_plane_rings = vec![OUTSIDE_HALF_DISK; n * n];
        half_plane_rings
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(y, row)| {
                let fy = frequency(y, n);
                for (x, ring) in row.iter_mut().enumerate() {
                    let fx = frequency(x, n);
                    let r = ring_of(fy, fx);
                    if r < n / 2 && (fy > 0 || (fy == 0 && fx > 0)) {
                        *ring = r as u16;
                    }
                }
            });
        let mut ring_counts = vec![0; n / 2];
        for &r in &half_plane_rings {
            if r != OUTSIDE_HALF_DISK {
                ring_counts[r as usize] += 2;
            }
        }
        ring_counts[0] += 1;
        Self {
            n,
            fft: FftPlanner::new().plan_fft_forward(n),
            half_plane_rings,
            ring_counts,
            spectrum: vec![Complex32::default(); n * n],
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// The FRC curve of an image of n x n counts in row-major order, split with `seed`: its value
    /// for each ring r < N / 2. The same counts and seed give the same curve.
    pub fn one_frc(&mut self, counts: &[i32], seed: u64) -> Result<Vec<f64>, ToUsizeError> {
        Ok(self
            .one_frc_ring_sums(counts, seed)?
            .iter()
            .map(RingSums::frc)
            .collect())
    }

    /// The ring sums behind `one_frc`.
    pub fn one_frc_ring_sums(
        &mut self,
        counts: &[i32],
        seed: u64,
    ) -> Result<Vec<RingSums>, ToUsizeError> {
        self.split_fft_rows(counts, seed)?;
        self.fft_columns();
        Ok(self.ring_sums())
    }

    fn split_rows(
        &mut self,
        counts: &[i32],
        seed: u64,
        transform: bool,
    ) -> Result<(), ToUsizeError> {
        assert_eq!(
            counts.len(),
            self.n * self.n,
            "counts must be an n x n image"
        );
        let rows = (SPLIT_TASK_COUNTS / self.n).max(1) * self.n;
        let splitter = LanesSplitter::<16>::new(seed, CpuFeatures::Detected);
        let fft = &self.fft;
        self.spectrum
            .par_chunks_mut(rows)
            .zip(counts.par_chunks(rows))
            .enumerate()
            .map_init(
                || vec![Complex32::default(); fft.get_inplace_scratch_len()],
                |scratch, (i, (z, c))| {
                    splitter
                        .split_chunk(i, c, z, |n, k| Complex32::new(k as f32, (n - k) as f32))?;
                    if transform {
                        fft.process_with_scratch(z, scratch);
                    }
                    Ok(())
                },
            )
            .collect()
    }

    /// The split alone, without the row FFTs: a step of `one_frc` for checks and benchmarks.
    pub fn split(&mut self, counts: &[i32], seed: u64) -> Result<(), ToUsizeError> {
        self.split_rows(counts, seed, false)
    }

    /// The split and the row FFTs, a few rows at a time while they are in cache.
    pub fn split_fft_rows(&mut self, counts: &[i32], seed: u64) -> Result<(), ToUsizeError> {
        self.split_rows(counts, seed, true)
    }

    /// The row FFTs alone, of whatever the spectrum holds.
    pub fn fft_rows(&mut self) {
        let fft = &self.fft;
        let rows = (SPLIT_TASK_COUNTS / self.n).max(1) * self.n;
        self.spectrum.par_chunks_mut(rows).for_each_init(
            || vec![Complex32::default(); fft.get_inplace_scratch_len()],
            |scratch, rows| fft.process_with_scratch(rows, scratch),
        );
    }

    /// The FFT of every column, in place: each task copies up to `COLUMN_BLOCK` columns into a
    /// buffer that stays in cache, transforms them as rows, and copies them back.
    pub fn fft_columns(&mut self) {
        let n = self.n;
        let fft = &self.fft;
        let z = SpectrumColumns(self.spectrum.as_mut_ptr());
        let stride = n + COLUMN_PAD;
        (0..n.div_ceil(COLUMN_BLOCK)).into_par_iter().for_each_init(
            || {
                let columns = vec![Complex32::default(); COLUMN_BLOCK * stride];
                (
                    columns,
                    vec![Complex32::default(); fft.get_inplace_scratch_len()],
                )
            },
            |(columns, scratch), block| {
                let p = z.ptr();
                let x0 = block * COLUMN_BLOCK;
                let width = COLUMN_BLOCK.min(n - x0);
                // In tiles of COLUMN_BLOCK rows: the tile's rows come into L1 once, and each row
                // of the buffer gets consecutive values, whole cache lines.
                for y0 in (0..n).step_by(COLUMN_BLOCK) {
                    prefetch_column_tile(p, n, x0, y0 + PREFETCH_ROWS);
                    for c in 0..width {
                        for y in y0..(y0 + COLUMN_BLOCK).min(n) {
                            // SAFETY: y * n + x0 + c < n * n, and only this task touches columns
                            // x0..x0 + width.
                            columns[c * stride + y] = unsafe { *p.add(y * n + x0 + c) };
                        }
                    }
                }
                for c in 0..width {
                    fft.process_with_scratch(&mut columns[c * stride..c * stride + n], scratch);
                }
                for y0 in (0..n).step_by(COLUMN_BLOCK) {
                    prefetch_column_tile(p, n, x0, y0 + PREFETCH_ROWS);
                    for y in y0..(y0 + COLUMN_BLOCK).min(n) {
                        for c in 0..width {
                            // SAFETY: as above.
                            unsafe { *p.add(y * n + x0 + c) = columns[c * stride + y] };
                        }
                    }
                }
            },
        );
    }

    /// The ring sums of the spectrum, over the half plane, doubled, plus the origin.
    pub fn ring_sums(&self) -> Vec<RingSums> {
        let n = self.n;
        let spectrum = &self.spectrum;
        let terms = |y: usize, x: usize| {
            let z = spectrum[y * n + x];
            let zm = spectrum[((n - y) % n) * n + (n - x) % n].conj();
            let f1 = (z + zm) * 0.5;
            let f2 = (z - zm) * Complex32::new(0.0, -0.5);
            [
                (f1 * f2.conj()).re as f64,
                f1.norm_sqr() as f64,
                f2.norm_sqr() as f64,
            ]
        };
        let add = |a: &mut [f64; 3], b: [f64; 3]| {
            for i in 0..3 {
                a[i] += b[i];
            }
        };
        let sums = self
            .half_plane_rings
            .par_chunks(n)
            .enumerate()
            .fold(
                || vec![[0f64; 3]; n / 2],
                |mut acc, (y, rings)| {
                    for (x, &r) in rings.iter().enumerate() {
                        if r != OUTSIDE_HALF_DISK {
                            add(&mut acc[r as usize], terms(y, x));
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![[0f64; 3]; n / 2],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        add(a, b);
                    }
                    a
                },
            );
        let origin = terms(0, 0);
        sums.iter()
            .zip(&self.ring_counts)
            .enumerate()
            .map(|(r, (s, &count))| {
                let [c, p1, p2] = s.map(|v| 2.0 * v);
                let o = if r == 0 { origin } else { [0.0; 3] };
                RingSums {
                    correlation: c + o[0],
                    power1: p1 + o[1],
                    power2: p2 + o[2],
                    count,
                }
            })
            .collect()
    }
}

/// Spatial frequencies (u, v, amplitude) of the cosine patterns in `pattern_image`.
pub const PATTERN_IMAGE_WAVES: [(f64, f64, f64); 4] = [
    (3.0, 5.0, 0.4),
    (17.0, -9.0, 0.2),
    (40.0, 33.0, 0.1),
    (120.0, 7.0, 0.05),
];

/// A synthetic image for checks and benchmarks: Poisson counts around `mean` times one plus the
/// cosines of `PATTERN_IMAGE_WAVES`, whose rings (6, 19, 52 and 120, for n > 240) have an FRC near
/// 1, while the noise elsewhere has an FRC near 0.
pub fn pattern_image(n: usize, mean: f64, seed: u64) -> Vec<i32> {
    let mut image = vec![0; n * n];
    image.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(
            seed ^ (y as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        for (x, v) in row.iter_mut().enumerate() {
            let phase = |u: f64, w: f64| {
                2.0 * std::f64::consts::PI * (u * x as f64 + w * y as f64) / n as f64
            };
            let waves: f64 = PATTERN_IMAGE_WAVES
                .iter()
                .map(|&(u, w, a)| a * phase(u, w).cos())
                .sum();
            *v = Poisson::new(mean * (1.0 + waves)).unwrap().sample(&mut rng) as i32;
        }
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::num_complex::Complex64;

    /// The 2D DFT by its definition, in f64.
    fn naive_dft2(z: &[Complex64], n: usize) -> Vec<Complex64> {
        let w = |k: usize| {
            Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64)
        };
        let mut rows = vec![Complex64::default(); n * n];
        for y in 0..n {
            for u in 0..n {
                rows[y * n + u] = (0..n).map(|x| z[y * n + x] * w(u * x % n)).sum();
            }
        }
        let mut out = vec![Complex64::default(); n * n];
        for v in 0..n {
            for u in 0..n {
                out[v * n + u] = (0..n).map(|y| rows[y * n + u] * w(v * y % n)).sum();
            }
        }
        out
    }

    /// The ring sums of two real images by their definition, over the whole plane.
    fn naive_ring_sums(h1: &[f64], h2: &[f64], n: usize) -> Vec<RingSums> {
        let as_complex = |h: &[f64]| {
            h.iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect::<Vec<_>>()
        };
        let (f1, f2) = (
            naive_dft2(&as_complex(h1), n),
            naive_dft2(&as_complex(h2), n),
        );
        let mut sums = vec![RingSums::default(); n / 2];
        for y in 0..n {
            for x in 0..n {
                let r = ring_of(frequency(y, n), frequency(x, n));
                if r < n / 2 {
                    let (a, b) = (f1[y * n + x], f2[y * n + x]);
                    let s = &mut sums[r];
                    s.correlation += (a * b.conj()).re;
                    s.power1 += a.norm_sqr();
                    s.power2 += b.norm_sqr();
                    s.count += 1;
                }
            }
        }
        sums
    }

    #[test]
    fn ring_sums_match_their_definition() {
        for n in [16, 21, 24, 30] {
            let counts = pattern_image(n, 50.0, n as u64);
            let mut plan = FrcPlan::new(n);
            plan.split(&counts, 7).unwrap();
            let halves = plan.spectrum.clone();
            plan.fft_rows();
            plan.fft_columns();
            let got = plan.ring_sums();
            let h1: Vec<f64> = halves.iter().map(|z| z.re as f64).collect();
            let h2: Vec<f64> = halves.iter().map(|z| z.im as f64).collect();
            assert!(
                h1.iter()
                    .zip(&h2)
                    .zip(&counts)
                    .all(|((a, b), &c)| a + b == c as f64)
            );
            let want = naive_ring_sums(&h1, &h2, n);
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.count, w.count, "n = {n}, ring {r}");
                let scale = w.power1.max(w.power2);
                for (gv, wv) in [
                    (g.correlation, w.correlation),
                    (g.power1, w.power1),
                    (g.power2, w.power2),
                ] {
                    assert!(
                        (gv - wv).abs() <= 1e-5 * scale,
                        "n = {n}, ring {r}: {gv} against {wv}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_split_matches_its_steps() {
        let n = 64;
        let counts = pattern_image(n, 80.0, 1);
        let mut plan = FrcPlan::new(n);
        plan.split(&counts, 3).unwrap();
        plan.fft_rows();
        let stepwise = plan.spectrum.clone();
        plan.split_fft_rows(&counts, 3).unwrap();
        assert_eq!(stepwise, plan.spectrum);
    }

    #[test]
    fn pattern_rings_correlate() {
        let n = 512;
        let mut plan = FrcPlan::new(n);
        let frc = plan.one_frc(&pattern_image(n, 100.0, 2), 5).unwrap();
        // The weakest wave, at ring 120, stands about 90 times above its ring's noise at this
        // size, and its cross terms with the noise move the FRC by a few percent.
        for r in [6, 19, 52, 120] {
            assert!(frc[r] > 0.9, "ring {r}: {}", frc[r]);
        }
        let mut noise: Vec<f64> = frc[130..].iter().map(|v| v.abs()).collect();
        noise.sort_by(f64::total_cmp);
        assert!(
            noise[noise.len() / 2] < 0.05,
            "median |FRC| of the noise rings {}",
            noise[noise.len() / 2]
        );
    }

    #[test]
    fn same_seed_same_curve() {
        let n = 100;
        let counts = pattern_image(n, 30.0, 3);
        let mut plan = FrcPlan::new(n);
        assert_eq!(
            plan.one_frc(&counts, 11).unwrap(),
            plan.one_frc(&counts, 11).unwrap()
        );
        assert_ne!(
            plan.one_frc(&counts, 11).unwrap(),
            plan.one_frc(&counts, 12).unwrap()
        );
    }

    #[test]
    fn negative_counts_are_refused() {
        let n = 32;
        let mut counts = pattern_image(n, 10.0, 4);
        counts[500] = -1;
        assert!(FrcPlan::new(n).one_frc(&counts, 1).is_err());
    }
}
