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
//! - The split writes z a block of rows at a time and transforms those rows while they are in
//!   cache.
//! - The spectrum is stored in panels of 16 columns, so the column FFTs read and write memory in
//!   order, a panel at a time, without a transpose pass.
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

/// Columns of one panel of the spectrum. The spectrum is stored in panels of `PANEL` columns:
/// panel p holds columns PANEL p .. PANEL p + w (w = PANEL, but for the last panel the rest),
/// row after row, so that the column FFTs read and write memory in order. 16 complex f32 are two
/// cache lines.
const PANEL: usize = 16;

/// Counts that one task of the split samples and transforms: whole rows, 16 of them at 4096,
/// so that each task writes 2 KiB at a time into every panel.
const SPLIT_TASK_COUNTS: usize = 65536;

/// Complex f32s between the starts of two rows of the column buffer beyond the row itself: one
/// cache line, so that rows do not start a multiple of 4 KiB apart, where the values that one
/// row of a panel adds to them would all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// Where entry (y, x) of an n x n image lies in the panel layout.
fn panel_index(n: usize, y: usize, x: usize) -> usize {
    let p = x / PANEL;
    let width = PANEL.min(n - p * PANEL);
    p * PANEL * n + y * width + (x - p * PANEL)
}

/// What `one_frc` needs for images of one size, made once like an FFT plan.
pub struct FrcPlan {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    /// In the panel layout, for each frequency of the half plane fy > 0 or (fy = 0, fx > 0) with
    /// r = round(|f|) < N/2, its ring r; every other entry is `OUTSIDE_HALF_DISK`.
    half_plane_rings: Vec<u16>,
    ring_counts: Vec<u64>,
    /// In the panel layout, the halves as z = k + i (n - k), then the spectrum Z.
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

/// The rows that hold the half plane fy > 0 or fy = 0: 0 ..= (n - 1) / 2.
fn half_plane_rows(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// `FrcPlan::spectrum` shared by the tasks of the split, which each write their own rows of every
/// panel only.
#[derive(Clone, Copy)]
struct SpectrumRows(*mut Complex32);
// SAFETY: the tasks that share it write disjoint rows.
unsafe impl Send for SpectrumRows {}
unsafe impl Sync for SpectrumRows {}

impl SpectrumRows {
    fn ptr(self) -> *mut Complex32 {
        self.0
    }
}

/// How a task of the split leaves its rows.
#[derive(Clone, Copy, PartialEq)]
enum RowStep {
    Split,
    SplitAndTransform,
    /// No split: transform the rows that the spectrum holds.
    Transform,
}

impl FrcPlan {
    /// A plan for images of n x n counts.
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 2 && n / 2 < OUTSIDE_HALF_DISK as usize,
            "unsupported image size {n}"
        );
        let mut half_plane_rings = vec![OUTSIDE_HALF_DISK; n * n];
        let mut ring_counts = vec![0; n / 2];
        for y in 0..n {
            let fy = frequency(y, n);
            for x in 0..n {
                let fx = frequency(x, n);
                let r = ring_of(fy, fx);
                if r < n / 2 && (fy > 0 || (fy == 0 && fx > 0)) {
                    half_plane_rings[panel_index(n, y, x)] = r as u16;
                    ring_counts[r] += 2;
                }
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

    /// Runs `step` on blocks of rows: each task works on its rows in a buffer of its own, in row
    /// order, and copies them into every panel of the spectrum at the end.
    fn rows_through_buffer(
        &mut self,
        counts: Option<&[i32]>,
        seed: u64,
        step: RowStep,
    ) -> Result<(), ToUsizeError> {
        let n = self.n;
        if let Some(counts) = counts {
            assert_eq!(counts.len(), n * n, "counts must be an n x n image");
        }
        let rows = (SPLIT_TASK_COUNTS / n).max(1);
        let splitter = LanesSplitter::<16>::new(seed, CpuFeatures::Detected);
        let fft = &self.fft;
        let spectrum = SpectrumRows(self.spectrum.as_mut_ptr());
        (0..n.div_ceil(rows))
            .into_par_iter()
            .map_init(
                || {
                    (
                        vec![Complex32::default(); rows * n],
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                    )
                },
                |(buffer, scratch), task| {
                    let y0 = task * rows;
                    let height = rows.min(n - y0);
                    let buffer = &mut buffer[..height * n];
                    let p = spectrum.ptr();
                    let panels = n.div_ceil(PANEL);
                    if step == RowStep::Transform {
                        for panel in 0..panels {
                            let (x0, width) = (panel * PANEL, PANEL.min(n - panel * PANEL));
                            for r in 0..height {
                                // SAFETY: the run lies within the spectrum, in rows y0..y0 + height,
                                // which only this task touches.
                                let run = unsafe {
                                    std::slice::from_raw_parts(
                                        p.add(panel_index(n, y0 + r, x0)),
                                        width,
                                    )
                                };
                                buffer[r * n + x0..r * n + x0 + width].copy_from_slice(run);
                            }
                        }
                    } else {
                        let counts = &counts.unwrap()[y0 * n..(y0 + height) * n];
                        splitter.split_chunk(task, counts, buffer, |n, k| {
                            Complex32::new(k as f32, (n - k) as f32)
                        })?;
                    }
                    if step != RowStep::Split {
                        fft.process_with_scratch(buffer, scratch);
                    }
                    for panel in 0..panels {
                        let (x0, width) = (panel * PANEL, PANEL.min(n - panel * PANEL));
                        // SAFETY: as above; rows y0..y0 + height of one panel are one run.
                        let run = unsafe {
                            std::slice::from_raw_parts_mut(
                                p.add(panel_index(n, y0, x0)),
                                height * width,
                            )
                        };
                        for r in 0..height {
                            run[r * width..(r + 1) * width]
                                .copy_from_slice(&buffer[r * n + x0..r * n + x0 + width]);
                        }
                    }
                    Ok(())
                },
            )
            .collect()
    }

    /// The split alone, without the row FFTs: a step of `one_frc` for checks and benchmarks.
    pub fn split(&mut self, counts: &[i32], seed: u64) -> Result<(), ToUsizeError> {
        self.rows_through_buffer(Some(counts), seed, RowStep::Split)
    }

    /// The split and the row FFTs, a block of rows at a time while they are in cache.
    pub fn split_fft_rows(&mut self, counts: &[i32], seed: u64) -> Result<(), ToUsizeError> {
        self.rows_through_buffer(Some(counts), seed, RowStep::SplitAndTransform)
    }

    /// The row FFTs alone, of whatever the spectrum holds.
    pub fn fft_rows(&mut self) {
        self.rows_through_buffer(None, 0, RowStep::Transform)
            .unwrap();
    }

    /// The FFT of every column, in place, a panel at a time: each task copies its panel's columns
    /// into a buffer that stays in cache, transforms them, and copies them back.
    pub fn fft_columns(&mut self) {
        let n = self.n;
        let fft = &self.fft;
        let stride = n + COLUMN_PAD;
        self.spectrum.par_chunks_mut(PANEL * n).for_each_init(
            || {
                (
                    vec![Complex32::default(); PANEL * stride],
                    vec![Complex32::default(); fft.get_inplace_scratch_len()],
                )
            },
            |(columns, scratch), panel| {
                let width = panel.len() / n;
                for (y, row) in panel.chunks_exact(width).enumerate() {
                    for (c, &v) in row.iter().enumerate() {
                        columns[c * stride + y] = v;
                    }
                }
                for c in 0..width {
                    fft.process_with_scratch(&mut columns[c * stride..c * stride + n], scratch);
                }
                for (y, row) in panel.chunks_exact_mut(width).enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        *v = columns[c * stride + y];
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
            let z = spectrum[panel_index(n, y, x)];
            let zm = spectrum[panel_index(n, (n - y) % n, (n - x) % n)].conj();
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
        let rows = half_plane_rows(n);
        let sums = self
            .half_plane_rings
            .par_chunks(PANEL * n)
            .enumerate()
            .fold(
                || vec![[0f64; 3]; n / 2],
                |mut acc, (panel, rings)| {
                    let width = rings.len() / n;
                    let x0 = panel * PANEL;
                    for (y, row) in rings[..rows * width].chunks_exact(width).enumerate() {
                        for (c, &r) in row.iter().enumerate() {
                            if r != OUTSIDE_HALF_DISK {
                                add(&mut acc[r as usize], terms(y, x0 + c));
                            }
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

    /// The spectrum, or after `split` the halves, in row-major order.
    pub fn spectrum_row_major(&self) -> Vec<Complex32> {
        let n = self.n;
        (0..n * n)
            .map(|i| self.spectrum[panel_index(n, i / n, i % n)])
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
            let halves = plan.spectrum_row_major();
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
