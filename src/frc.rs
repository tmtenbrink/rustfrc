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
//! - The spectrum is stored in panels of 8 columns and their 8 mirrors, so the column FFTs read
//!   memory in order, a panel at a time, without a transpose pass, and each panel holds both
//!   Z(f) and Z(-f) for its ring terms. The ring sums are taken while the panel is in cache, and
//!   the column FFTs are never written back.
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

/// Pairs of mirrored columns, x and n - x, that one panel of the spectrum holds. Each panel holds
/// the 8 columns 8q + 1 ..= 8q + 8 followed by their mirrors n - 8q - 8 ..= n - 8q - 1, row after
/// row, so the column FFTs read and write memory in order, and the ring sums, which need Z(f) and
/// Z(-f), find both in the same panel. The columns that are their own mirror, 0 and for even n
/// also n / 2, share a last panel.
const PAIRS_PER_PANEL: usize = 8;

/// Counts that one task of the split samples and transforms: whole rows, 16 of them at 4096,
/// so that each task writes runs of 16 rows into every panel.
const SPLIT_TASK_COUNTS: usize = 65536;

/// Complex f32s between the starts of two rows of the column buffer beyond the row itself: one
/// cache line, so that rows do not start a multiple of 4 KiB apart, where the values that one
/// row of a panel adds to them would all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// Columns that one panel of the spectrum holds, where it starts, and for each of its columns the
/// position of the mirror column in the panel.
struct Panel {
    start: usize,
    columns: Vec<usize>,
    mirror_slots: Vec<usize>,
    /// The columns as runs of consecutive ones: (first column, its position in the panel, length).
    runs: Vec<(usize, usize, usize)>,
}

/// The runs of consecutive columns in `columns`, for copying a panel's row in a few pieces.
fn column_runs(columns: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (c, &x) in columns.iter().enumerate() {
        match runs.last_mut() {
            Some((first, _, len)) if *first + *len == x => *len += 1,
            _ => runs.push((x, c, 1)),
        }
    }
    runs
}

/// The panels of an n x n spectrum, and where each column lies in them.
struct PanelLayout {
    panels: Vec<Panel>,
    /// For each column: its panel, and its position in the panel.
    column_slots: Vec<(usize, usize)>,
}

impl PanelLayout {
    fn new(n: usize) -> Self {
        let pairs = (n - 1) / 2;
        let mut groups: Vec<Vec<usize>> = (0..pairs.div_ceil(PAIRS_PER_PANEL))
            .map(|q| {
                let first = q * PAIRS_PER_PANEL + 1;
                let last = (first + PAIRS_PER_PANEL - 1).min(pairs);
                (first..=last).chain(n - last..=n - first).collect()
            })
            .collect();
        groups.push(if n.is_multiple_of(2) {
            vec![0, n / 2]
        } else {
            vec![0]
        });
        let mut column_slots = vec![(0, 0); n];
        let mut start = 0;
        let panels = groups
            .into_iter()
            .enumerate()
            .map(|(p, columns)| {
                for (c, &x) in columns.iter().enumerate() {
                    column_slots[x] = (p, c);
                }
                let mirror_slots = columns
                    .iter()
                    .map(|&x| columns.iter().position(|&m| m == mirrored(x, n)).unwrap())
                    .collect();
                let runs = column_runs(&columns);
                let panel = Panel {
                    start,
                    columns,
                    mirror_slots,
                    runs,
                };
                start += panel.columns.len() * n;
                panel
            })
            .collect();
        Self {
            panels,
            column_slots,
        }
    }

    fn index(&self, y: usize, x: usize) -> usize {
        let (p, c) = self.column_slots[x];
        let panel = &self.panels[p];
        panel.start + y * panel.columns.len() + c
    }
}

/// What `one_frc` needs for images of one size, made once like an FFT plan.
pub struct FrcPlan {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    layout: PanelLayout,
    ring_counts: Vec<u64>,
    /// In the panel layout, the halves as z = k + i (n - k), then the spectrum Z after the row
    /// FFTs. `fft_columns` alone transforms the columns in place; `one_frc` does not write them.
    spectrum: Vec<Complex32>,
}

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

/// The index of frequency -f for the index i of frequency f, in a DFT of length n: (n - i) % n,
/// without the division that % by a run-time n costs.
fn mirrored(i: usize, n: usize) -> usize {
    if i == 0 { 0 } else { n - i }
}

/// The rows that hold the half plane fy > 0 or fy = 0: 0 ..= (n - 1) / 2.
fn half_plane_rows(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// Copies `src` to `dst` with streaming stores where `dst` allows them: they write whole cache
/// lines without reading them from memory first, which a normal store does. The caller fences.
fn copy_streaming(dst: &mut [Complex32], src: &[Complex32]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if (dst.as_ptr() as usize).is_multiple_of(16) && dst.len().is_multiple_of(2) {
        use std::arch::x86_64::{_mm_loadu_ps, _mm_stream_ps};
        let (d, s) = (dst.as_mut_ptr() as *mut f32, src.as_ptr() as *const f32);
        for i in (0..2 * dst.len()).step_by(4) {
            // SAFETY: i + 4 <= 2 len for both, d + i is 16-byte aligned, and SSE is part of x86-64.
            unsafe { _mm_stream_ps(d.add(i), _mm_loadu_ps(s.add(i))) };
        }
        return;
    }
    dst.copy_from_slice(src)
}

/// Ring sums of the half plane, not yet doubled, in slots 0..n/2, and the origin's terms in slot
/// n / 2.
type HalfPlaneSums = Vec<[f64; 3]>;

fn add_terms(a: &mut [f64; 3], b: [f64; 3]) {
    for i in 0..3 {
        a[i] += b[i];
    }
}

/// F1 conj(F2), |F1|^2 and |F2|^2 from Z(f) and Z(-f).
fn ring_terms(z: Complex32, z_mirror: Complex32) -> [f64; 3] {
    let zm = z_mirror.conj();
    let f1 = (z + zm) * 0.5;
    let f2 = (z - zm) * Complex32::new(0.0, -0.5);
    [
        (f1 * f2.conj()).re as f64,
        f1.norm_sqr() as f64,
        f2.norm_sqr() as f64,
    ]
}

/// Adds the terms of the frequencies of `panel` in the half plane to `sums`, where `at(c, y)` is
/// the spectrum at row y of the panel's column c.
fn add_panel_ring_terms(
    n: usize,
    panel: &Panel,
    at: impl Fn(usize, usize) -> Complex32,
    sums: &mut HalfPlaneSums,
) {
    let half = n / 2;
    for (c, &x) in panel.columns.iter().enumerate() {
        let fx = frequency(x, n);
        let m = panel.mirror_slots[c];
        if x == 0 {
            add_terms(&mut sums[half], ring_terms(at(c, 0), at(c, 0)));
        }
        // Row 0 belongs to the half plane only where fx > 0.
        let first = if fx > 0 { 0 } else { 1 };
        // Down the column |f|^2 only grows, and ring r holds r^2 - r + 1 <= |f|^2 <= r^2 + r: the
        // ring follows with a comparison per row, and a run of rows in one ring is summed apart.
        let mut r = ring_of(first as i64, fx) as i64;
        let mut run = [0f64; 3];
        for y in first..half_plane_rows(n) {
            let d2 = fx * fx + (y * y) as i64;
            while d2 > r * r + r {
                if r < half as i64 {
                    add_terms(&mut sums[r as usize], run);
                }
                run = [0.0; 3];
                r += 1;
            }
            if r >= half as i64 {
                break;
            }
            add_terms(&mut run, ring_terms(at(c, y), at(m, mirrored(y, n))));
        }
        if r < half as i64 {
            add_terms(&mut sums[r as usize], run);
        }
    }
}

/// The spectrum, split into its panels for the tasks of the column pass.
fn split_into_panels<'a>(
    mut spectrum: &'a mut [Complex32],
    layout: &PanelLayout,
    n: usize,
) -> Vec<&'a mut [Complex32]> {
    layout
        .panels
        .iter()
        .map(|panel| {
            let (this, rest) = std::mem::take(&mut spectrum).split_at_mut(panel.columns.len() * n);
            spectrum = rest;
            this
        })
        .collect()
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

/// What the column pass does with each panel after its column FFTs.
#[derive(Clone, Copy, PartialEq)]
enum ColumnStep {
    /// Write the transformed columns back to the spectrum.
    WriteBack,
    /// Add the panel's ring terms to the ring sums, and write nothing back.
    RingSums,
}

impl FrcPlan {
    /// A plan for images of n x n counts.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "unsupported image size {n}");
        let mut ring_counts = vec![0; n / 2];
        for y in 0..n {
            for x in 0..n {
                let r = ring_of(frequency(y, n), frequency(x, n));
                if r < n / 2 {
                    ring_counts[r] += 1;
                }
            }
        }
        Self {
            n,
            fft: FftPlanner::new().plan_fft_forward(n),
            layout: PanelLayout::new(n),
            ring_counts,
            spectrum: vec![Complex32::default(); n * n],
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// The number of frequencies in each ring r < N / 2.
    pub fn ring_counts(&self) -> &[u64] {
        &self.ring_counts
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
        Ok(self.fft_columns_ring_sums())
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
        let (fft, layout) = (&self.fft, &self.layout);
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
                    if step == RowStep::Transform {
                        for panel in &layout.panels {
                            let width = panel.columns.len();
                            // SAFETY: rows y0..y0 + height of a panel are one run within the
                            // spectrum, which only this task touches.
                            let run = unsafe {
                                std::slice::from_raw_parts(
                                    p.add(panel.start + y0 * width),
                                    height * width,
                                )
                            };
                            for (r, row) in run.chunks_exact(width).enumerate() {
                                for &(x, c, len) in &panel.runs {
                                    buffer[r * n + x..r * n + x + len]
                                        .copy_from_slice(&row[c..c + len]);
                                }
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
                    let mut row_of_panel = [Complex32::default(); 2 * PAIRS_PER_PANEL];
                    for panel in &layout.panels {
                        let width = panel.columns.len();
                        // SAFETY: as above.
                        let run = unsafe {
                            std::slice::from_raw_parts_mut(
                                p.add(panel.start + y0 * width),
                                height * width,
                            )
                        };
                        for (r, out) in run.chunks_exact_mut(width).enumerate() {
                            for &(x, c, len) in &panel.runs {
                                row_of_panel[c..c + len]
                                    .copy_from_slice(&buffer[r * n + x..r * n + x + len]);
                            }
                            copy_streaming(out, &row_of_panel[..width]);
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: SSE is part of x86-64. The fence orders the streaming stores before
                    // whatever reads the spectrum next.
                    unsafe {
                        std::arch::x86_64::_mm_sfence()
                    };
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

    /// The column FFTs, a panel at a time: each task copies its panel's columns into a buffer that
    /// stays in cache and transforms them there, then does `step` with them.
    fn columns_through_buffer(&mut self, step: ColumnStep) -> HalfPlaneSums {
        let n = self.n;
        let (fft, layout) = (&self.fft, &self.layout);
        let stride = n + COLUMN_PAD;
        let panels = split_into_panels(&mut self.spectrum, layout, n);
        panels
            .into_par_iter()
            .zip(&layout.panels)
            .fold(
                || {
                    let columns = vec![Complex32::default(); 2 * PAIRS_PER_PANEL * stride];
                    (
                        columns,
                        vec![Complex32::default(); fft.get_inplace_scratch_len()],
                        vec![[0f64; 3]; n / 2 + 1],
                    )
                },
                |(mut columns, mut scratch, mut sums), (data, panel)| {
                    let width = panel.columns.len();
                    for (y, row) in data.chunks_exact(width).enumerate() {
                        for (c, &v) in row.iter().enumerate() {
                            columns[c * stride + y] = v;
                        }
                    }
                    for c in 0..width {
                        fft.process_with_scratch(
                            &mut columns[c * stride..c * stride + n],
                            &mut scratch,
                        );
                    }
                    match step {
                        ColumnStep::WriteBack => {
                            for (y, row) in data.chunks_exact_mut(width).enumerate() {
                                for (c, v) in row.iter_mut().enumerate() {
                                    *v = columns[c * stride + y];
                                }
                            }
                        }
                        ColumnStep::RingSums => {
                            add_panel_ring_terms(
                                n,
                                panel,
                                |c, y| columns[c * stride + y],
                                &mut sums,
                            );
                        }
                    }
                    (columns, scratch, sums)
                },
            )
            .map(|(_, _, sums)| sums)
            .reduce(
                || vec![[0f64; 3]; n / 2 + 1],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        add_terms(a, b);
                    }
                    a
                },
            )
    }

    /// The column FFTs alone, written back to the spectrum, for checks and benchmarks.
    pub fn fft_columns(&mut self) {
        self.columns_through_buffer(ColumnStep::WriteBack);
    }

    /// The column FFTs and the ring sums in one pass over the spectrum, which it leaves as the
    /// row FFTs left it.
    pub fn fft_columns_ring_sums(&mut self) -> Vec<RingSums> {
        let sums = self.columns_through_buffer(ColumnStep::RingSums);
        self.full_ring_sums(&sums)
    }

    /// The ring sums of the spectrum after `fft_columns`, for checks and benchmarks.
    pub fn ring_sums(&self) -> Vec<RingSums> {
        let n = self.n;
        let sums = split_panels_ref(&self.spectrum, &self.layout, n)
            .into_par_iter()
            .zip(&self.layout.panels)
            .fold(
                || vec![[0f64; 3]; n / 2 + 1],
                |mut sums, (data, panel)| {
                    let width = panel.columns.len();
                    add_panel_ring_terms(n, panel, |c, y| data[y * width + c], &mut sums);
                    sums
                },
            )
            .reduce(
                || vec![[0f64; 3]; n / 2 + 1],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        add_terms(a, b);
                    }
                    a
                },
            );
        self.full_ring_sums(&sums)
    }

    /// The sums over whole rings from those over the half plane: doubled, plus the origin.
    fn full_ring_sums(&self, sums: &HalfPlaneSums) -> Vec<RingSums> {
        let half = self.n / 2;
        let origin = sums[half];
        sums[..half]
            .iter()
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
            .map(|i| self.spectrum[self.layout.index(i / n, i % n)])
            .collect()
    }
}

/// `split_into_panels` for reading.
fn split_panels_ref<'a>(
    mut spectrum: &'a [Complex32],
    layout: &PanelLayout,
    n: usize,
) -> Vec<&'a [Complex32]> {
    layout
        .panels
        .iter()
        .map(|panel| {
            let (this, rest) = spectrum.split_at(panel.columns.len() * n);
            spectrum = rest;
            this
        })
        .collect()
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
    fn fused_ring_sums_match_their_steps() {
        for n in [2, 3, 17, 48, 64] {
            let counts = pattern_image(n, 40.0, 9);
            let mut plan = FrcPlan::new(n);
            plan.split_fft_rows(&counts, 2).unwrap();
            let fused = plan.fft_columns_ring_sums();
            plan.fft_columns();
            let stepwise = plan.ring_sums();
            for (r, (f, s)) in fused.iter().zip(&stepwise).enumerate() {
                assert_eq!(f.count, s.count, "n = {n}, ring {r}");
                for (a, b) in [
                    (f.correlation, s.correlation),
                    (f.power1, s.power1),
                    (f.power2, s.power2),
                ] {
                    assert!(
                        (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                        "n = {n}, ring {r}: {a} against {b}"
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
