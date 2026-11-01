//! Fourier ring correlation of one image, as frc.one_frc computes it with method 1 (one binomial
//! split, the second half the remainder), on the GPU (`frc.wgsl`) and on the CPU (rustfrc's
//! split and rustfft). Checks the two against each other and times both.
//!
//!   cargo run --release --bin frc -- [N] [MEAN] [REPS]      defaults: 4096 100 10
//!
//! N must be a power of two of at least 512. The test image has Poisson counts around MEAN with
//! four cosine patterns, at rings 6, 19, 52 and 120, where the FRC should be close to 1; it is
//! close to 0 elsewhere. A ring r holds the frequencies f with round(|f|) = r; that DIPlib's
//! RadialSum, which frc uses, bins the same way is not checked here.

use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use rustfrc::binom_variants::{CpuFeatures, LanesSplitter};
use rustfrc_gpu::{copy_from_mapped, median_time, ms, open_gpu};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const SPLIT_THREADS: u32 = 1 << 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    threads: u32,
    seed_lo: u32,
    seed_hi: u32,
}

/// Sums over one ring: F1 conj(F2), |F1|^2, |F2|^2, and the number of frequencies.
type RingSums = [f64; 4];

fn frc_curve(rings: &[RingSums]) -> Vec<f64> {
    rings.iter().map(|[num, d1, d2, _]| num / (d1 * d2).sqrt()).collect()
}

/// A dispatch of the GPU pipeline; `submit` runs a list of them in order.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    /// The split and the FFT of every row in one pass, from counts to the spectrum buffer.
    SplitRowFft,
    Split,
    /// FFT of every row, from the halves buffer to the spectrum buffer.
    RowFft,
    /// From the spectrum buffer back to the halves buffer.
    Transpose,
    Rings,
}

const FFT_2D: [Stage; 3] = [Stage::RowFft, Stage::Transpose, Stage::RowFft];
const ALL_STAGES: [Stage; 4] = [Stage::SplitRowFft, Stage::Transpose, Stage::RowFft, Stage::Rings];

struct GpuFrc {
    device: wgpu::Device,
    queue: wgpu::Queue,
    split: wgpu::ComputePipeline,
    split_fft: wgpu::ComputePipeline,
    fft: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    rings: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    counts: wgpu::Buffer,
    /// The halves as z = k + i (n - k), written by the split.
    halves: wgpu::Buffer,
    ring_sums: wgpu::Buffer,
    ring_readback: wgpu::Buffer,
    /// Where the CPU writes the counts, which the GPU then copies to `counts`. It lives as long
    /// as the pipeline: `Queue::write_buffer` allocates new staging memory at every call.
    upload: wgpu::Buffer,
    upload_mapped: Cell<bool>,
    split_group: wgpu::BindGroup,
    split_fft_group: wgpu::BindGroup,
    /// Both row FFTs read `halves` and write the spectrum buffer; the transpose goes back.
    fft_group: wgpu::BindGroup,
    transpose_group: wgpu::BindGroup,
    rings_group: wgpu::BindGroup,
    n: u32,
}

impl GpuFrc {
    fn new(n: u32) -> Self {
        let (device, queue, limits) = open_gpu();
        let start = Instant::now();
        let len = (n * n) as u64;
        assert!(n.is_power_of_two() && n >= 512, "N must be a power of two of at least 512");
        assert!(len * 8 <= limits.max_storage_buffer_binding_size, "N too large for one binding");
        assert!(
            n * 8 <= limits.max_compute_workgroup_storage_size,
            "a row of {n} does not fit in workgroup memory ({} bytes)",
            limits.max_compute_workgroup_storage_size
        );
        let source = include_str!("frc.wgsl")
            .replace("__N__", &n.to_string())
            .replace("__LOG2_N__", &n.trailing_zeros().to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frc.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (split, split_fft, fft, transpose, rings) = (
            pipeline("split"),
            pipeline("split_fft_rows"),
            pipeline("fft_rows"),
            pipeline("transpose"),
            pipeline("rings"),
        );

        use wgpu::BufferUsages as U;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let params = buffer("params", 16, U::UNIFORM | U::COPY_DST);
        let counts = buffer("counts", len * 4, U::STORAGE | U::COPY_DST);
        let upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload"),
            size: len * 4,
            usage: U::MAP_WRITE | U::COPY_SRC,
            mapped_at_creation: true,
        });
        let halves = buffer("halves", len * 8, U::STORAGE | U::COPY_DST);
        let spectrum = buffer("spectrum", len * 8, U::STORAGE);
        let ring_bytes = (n as u64 / 2) * 16;
        let ring_sums = buffer("ring sums", ring_bytes, U::STORAGE | U::COPY_SRC);
        let ring_readback = buffer("ring readback", ring_bytes, U::MAP_READ | U::COPY_DST);
        let twiddles: Vec<[f32; 2]> = (0..n / 2)
            .map(|m| {
                let angle = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        let twiddles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddles"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: U::STORAGE,
        });

        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<_> = entries
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let split_group = group(&split, &[(0, &params), (1, &counts), (2, &halves)]);
        let split_fft_group = group(&split_fft, &[(0, &params), (1, &counts), (4, &spectrum), (5, &twiddles)]);
        let fft_group = group(&fft, &[(3, &halves), (4, &spectrum), (5, &twiddles)]);
        let transpose_group = group(&transpose, &[(8, &spectrum), (9, &halves)]);
        let rings_group = group(&rings, &[(6, &spectrum), (7, &ring_sums)]);
        let gpu = Self {
            device,
            queue,
            split,
            split_fft,
            fft,
            transpose,
            rings,
            params,
            counts,
            halves,
            ring_sums,
            ring_readback,
            upload,
            upload_mapped: Cell::new(true),
            split_group,
            split_fft_group,
            fft_group,
            transpose_group,
            rings_group,
            n,
        };
        gpu.wait();
        println!("setup: shaders, pipelines, buffers {}", ms(start.elapsed()));
        gpu
    }

    fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    fn upload_counts(&self, counts: &[u32], seed: u64) {
        let params = Params { len: self.n * self.n, threads: SPLIT_THREADS, seed_lo: seed as u32, seed_hi: (seed >> 32) as u32 };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        if !self.upload_mapped.get() {
            // Waits for the GPU to finish copying the previous counts out of it.
            self.upload.map_async(wgpu::MapMode::Write, .., |r| r.unwrap());
            self.wait();
        }
        self.upload.get_mapped_range_mut(..).unwrap().copy_from_slice(bytemuck::cast_slice(counts));
        self.upload.unmap();
        self.upload_mapped.set(false);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.upload, 0, &self.counts, 0, None);
        self.queue.submit([encoder.finish()]);
    }

    /// Puts halves split elsewhere in place of the split's output, to check the FFT and rings.
    fn upload_halves(&self, z: &[Complex32]) {
        // SAFETY: Complex<f32> is repr(C), two f32s with no padding, and any bytes are valid u8s.
        let bytes = unsafe { std::slice::from_raw_parts(z.as_ptr() as *const u8, std::mem::size_of_val(z)) };
        self.queue.write_buffer(&self.halves, 0, bytes);
    }

    fn submit(&self, stages: &[Stage], read_rings: bool) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for stage in stages {
                let (pipeline, group, workgroups) = match stage {
                    Stage::SplitRowFft => (&self.split_fft, &self.split_fft_group, (self.n, 1)),
                    Stage::Split => (&self.split, &self.split_group, (SPLIT_THREADS / WORKGROUP_SIZE, 1)),
                    Stage::RowFft => (&self.fft, &self.fft_group, (self.n, 1)),
                    Stage::Transpose => (&self.transpose, &self.transpose_group, (self.n / 32, self.n / 32)),
                    Stage::Rings => (&self.rings, &self.rings_group, (self.n / 2, 1)),
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
            }
        }
        if read_rings {
            encoder.copy_buffer_to_buffer(&self.ring_sums, 0, &self.ring_readback, 0, None);
        }
        self.queue.submit([encoder.finish()]);
    }

    fn read_rings(&self) -> Vec<RingSums> {
        self.ring_readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        self.wait();
        let mut sums = vec![[0f32; 4]; self.n as usize / 2];
        copy_from_mapped(bytemuck::cast_slice_mut(&mut sums), &self.ring_readback.get_mapped_range(..).unwrap());
        self.ring_readback.unmap();
        sums.iter().map(|s| s.map(|v| v as f64)).collect()
    }

    /// The whole computation: upload, split, FFT, rings, and the curve back.
    fn frc(&self, counts: &[u32], seed: u64) -> Vec<f64> {
        self.upload_counts(counts, seed);
        self.submit(&ALL_STAGES, true);
        frc_curve(&self.read_rings())
    }
}

/// The CPU version of the same steps, on every core, with its buffers kept between calls as
/// the GPU keeps its own.
struct CpuFrc {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    /// For each frequency of the half plane fy > 0 or (fy = 0, fx > 0) with r = round(|f|) < N/2,
    /// its ring r; every other entry is `OUTSIDE`. Made once per size, like the GPU's twiddles.
    half_plane_rings: Vec<u16>,
    ring_counts: Vec<f64>,
    /// The halves as z = k + i (n - k), then the spectrum Z.
    z: Vec<Complex32>,
}

/// Rows that one task of `split_fft_rows` handles: 16384 counts, a chunk of `LanesSplitter`.
fn rows_per_task(n: usize) -> usize {
    (16384 / n).max(1)
}

/// Columns that `fft_columns` gathers into one buffer: 16 complex f32 are two cache lines of a row.
const COLUMN_BLOCK: usize = 16;

/// Complex f32s between the starts of two rows of `fft_columns`' buffer beyond the row itself:
/// one cache line, so that rows do not start a multiple of 4 KiB apart, where the 16 values that
/// one image row adds to them would all fall in the same cache set.
const COLUMN_PAD: usize = 8;

/// `self.z` shared by the tasks of `fft_columns`, which each touch their own columns only.
#[derive(Clone, Copy)]
struct ColumnsOf(*mut Complex32);
// SAFETY: the tasks that share it read and write disjoint columns.
unsafe impl Send for ColumnsOf {}
unsafe impl Sync for ColumnsOf {}

/// Rows ahead of the current tile that the column copies prefetch. Reads 32 KiB apart defeat the
/// hardware prefetchers, which follow streams within a 4 KiB page, so without this every row
/// waits for memory in turn.
fn prefetch_rows() -> usize {
    std::env::var("FRC_PREFETCH_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(64)
}

/// Asks for the two cache lines of columns x0..x0 + 16 of rows y..y + 16, where they exist.
#[inline(always)]
fn prefetch_tile(p: *mut Complex32, n: usize, x0: usize, y: usize) {
    #[cfg(target_arch = "x86_64")]
    for row in y..(y + COLUMN_BLOCK).min(n) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: a prefetch does not access memory; the address is within the buffer anyway.
        unsafe {
            let line = p.add(row * n + x0) as *const i8;
            _mm_prefetch(line, _MM_HINT_T0);
            _mm_prefetch(line.add(64), _MM_HINT_T0);
        }
    }
}

impl ColumnsOf {
    fn ptr(self) -> *mut Complex32 {
        self.0
    }
}

const OUTSIDE: u16 = u16::MAX;

/// The signed frequency of index i of an FFT of length n.
fn frequency(i: usize, n: usize) -> i64 {
    if i < n / 2 { i as i64 } else { i as i64 - n as i64 }
}

impl CpuFrc {
    fn new(n: usize) -> Self {
        assert!(n / 2 < OUTSIDE as usize);
        let mut half_plane_rings = vec![OUTSIDE; n * n];
        let mut ring_counts = vec![0.0; n / 2];
        half_plane_rings.par_chunks_mut(n).enumerate().for_each(|(a, row)| {
            let fa = frequency(a, n);
            for (b, ring) in row.iter_mut().enumerate() {
                let fb = frequency(b, n);
                let r = ((fa * fa + fb * fb) as f64).sqrt().round() as usize;
                if r < n / 2 && (fa > 0 || (fa == 0 && fb > 0)) {
                    *ring = r as u16;
                }
            }
        });
        for a in 0..n {
            for b in 0..n {
                let (fa, fb) = (frequency(a, n), frequency(b, n));
                let r = ((fa * fa + fb * fb) as f64).sqrt().round() as usize;
                if r < n / 2 {
                    ring_counts[r] += 1.0;
                }
            }
        }
        Self {
            n,
            fft: FftPlanner::new().plan_fft_forward(n),
            half_plane_rings,
            ring_counts,
            z: vec![Complex32::default(); n * n],
        }
    }

    /// Splits rows of counts into `z`, and if `transform` is set, transforms each block of rows
    /// while it is still in cache: one pass over memory instead of three.
    fn split_rows(&mut self, counts: &[i32], seed: u64, transform: bool) {
        let rows = rows_per_task(self.n) * self.n;
        let splitter = LanesSplitter::<16>::new(seed, CpuFeatures::Detected);
        let fft = &self.fft;
        self.z.par_chunks_mut(rows).zip(counts.par_chunks(rows)).enumerate().for_each_init(
            || vec![Complex32::default(); fft.get_inplace_scratch_len()],
            |scratch, (i, (z, c))| {
                splitter.split_chunk(i, c, z, |n, k| Complex32::new(k as f32, (n - k) as f32)).unwrap();
                if transform {
                    fft.process_with_scratch(z, scratch);
                }
            },
        );
    }

    fn split(&mut self, counts: &[i32], seed: u64) {
        self.split_rows(counts, seed, false);
    }

    fn split_fft_rows(&mut self, counts: &[i32], seed: u64) {
        self.split_rows(counts, seed, true);
    }

    fn fft_rows(&mut self) {
        let fft = &self.fft;
        self.z.par_chunks_mut(rows_per_task(self.n) * self.n).for_each_init(
            || vec![Complex32::default(); fft.get_inplace_scratch_len()],
            |scratch, rows| fft.process_with_scratch(rows, scratch),
        );
    }

    /// The FFT of every column, in place: each task copies `COLUMN_BLOCK` columns into a buffer
    /// that stays in cache, transforms them as rows, and copies them back. No transpose pass.
    fn fft_columns(&mut self) {
        self.columns_through_cache(true);
    }

    /// `fft_columns`, or with `transform` unset only its copies, to time them apart.
    fn columns_through_cache(&mut self, transform: bool) {
        let n = self.n;
        assert_eq!(n % COLUMN_BLOCK, 0);
        let fft = &self.fft;
        let z = ColumnsOf(self.z.as_mut_ptr());
        let ahead = prefetch_rows();
        (0..n / COLUMN_BLOCK).into_par_iter().for_each_init(
            || {
                let columns = vec![Complex32::default(); COLUMN_BLOCK * (n + COLUMN_PAD)];
                (columns, vec![Complex32::default(); fft.get_inplace_scratch_len()])
            },
            |(columns, scratch), block| {
                let p = z.ptr();
                let x0 = block * COLUMN_BLOCK;
                let stride = n + COLUMN_PAD;
                // In tiles of COLUMN_BLOCK rows: the tile's rows come into L1 once, and each row of
                // the buffer gets COLUMN_BLOCK consecutive values, whole cache lines.
                for y0 in (0..n).step_by(COLUMN_BLOCK) {
                    prefetch_tile(p, n, x0, y0 + ahead);
                    for c in 0..COLUMN_BLOCK {
                        for y in y0..y0 + COLUMN_BLOCK {
                            // SAFETY: y * n + x0 + c < n * n, and only this task touches columns
                            // x0..x0 + COLUMN_BLOCK.
                            columns[c * stride + y] = unsafe { *p.add(y * n + x0 + c) };
                        }
                    }
                }
                if transform {
                    for c in 0..COLUMN_BLOCK {
                        fft.process_with_scratch(&mut columns[c * stride..c * stride + n], scratch);
                    }
                }
                for y0 in (0..n).step_by(COLUMN_BLOCK) {
                    prefetch_tile(p, n, x0, y0 + ahead);
                    for y in y0..y0 + COLUMN_BLOCK {
                        for c in 0..COLUMN_BLOCK {
                            // SAFETY: as above.
                            unsafe { *p.add(y * n + x0 + c) = columns[c * stride + y] };
                        }
                    }
                }
            },
        );
    }

    /// The ring sums of the spectrum in `self.z`, over the half plane, doubled, plus the origin.
    fn rings(&self) -> Vec<RingSums> {
        let n = self.n;
        let spectrum = &self.z;
        let terms = |a: usize, b: usize| {
            let z = spectrum[a * n + b];
            let zm = spectrum[((n - a) % n) * n + (n - b) % n].conj();
            let f1 = (z + zm) * 0.5;
            let f2 = (z - zm) * Complex32::new(0.0, -0.5);
            [(f1 * f2.conj()).re as f64, f1.norm_sqr() as f64, f2.norm_sqr() as f64]
        };
        let mut sums = self
            .half_plane_rings
            .par_chunks(n)
            .enumerate()
            .fold(
                || vec![[0f64; 3]; n / 2],
                |mut acc, (a, rings)| {
                    for (b, &r) in rings.iter().enumerate() {
                        if r != OUTSIDE {
                            let t = terms(a, b);
                            let s = &mut acc[r as usize];
                            for i in 0..3 {
                                s[i] += t[i];
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![[0f64; 3]; n / 2],
                |mut x, y| {
                    for (x, y) in x.iter_mut().zip(y) {
                        for i in 0..3 {
                            x[i] += y[i];
                        }
                    }
                    x
                },
            );
        for s in &mut sums {
            for v in s.iter_mut() {
                *v *= 2.0;
            }
        }
        let origin = terms(0, 0);
        for i in 0..3 {
            sums[0][i] += origin[i];
        }
        sums.iter().zip(&self.ring_counts).map(|(s, &c)| [s[0], s[1], s[2], c]).collect()
    }

    fn frc(&mut self, counts: &[i32], seed: u64) -> Vec<f64> {
        self.split_fft_rows(counts, seed);
        self.fft_columns();
        frc_curve(&self.rings())
    }
}

/// Poisson counts around `mean` with four cosine patterns, so that the FRC is close to 1 at
/// rings 6, 19, 52 and 120.
fn test_image(n: usize, mean: f64) -> Vec<i32> {
    let patterns = [(3.0, 5.0, 0.4), (17.0, -9.0, 0.2), (40.0, 33.0, 0.1), (120.0, 7.0, 0.05)];
    let mut image = vec![0; n * n];
    image.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(y as u64);
        for (x, v) in row.iter_mut().enumerate() {
            let phase = |u: f64, w: f64| 2.0 * std::f64::consts::PI * (u * x as f64 + w * y as f64) / n as f64;
            let lambda = mean * (1.0 + patterns.iter().map(|&(u, w, a)| a * phase(u, w).cos()).sum::<f64>());
            *v = Poisson::new(lambda).unwrap().sample(&mut rng) as i32;
        }
    });
    image
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu_only = args.iter().any(|a| a == "--cpu-only");
    let numbers: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let n: usize = numbers.first().map_or(4096, |s| s.parse().unwrap());
    let mean: f64 = numbers.get(1).map_or(100.0, |s| s.parse().unwrap());
    let reps: usize = numbers.get(2).map_or(10, |s| s.parse().unwrap());
    println!("image: {n} x {n}, Poisson mean {mean} with patterns, {reps} reps, median times\n");
    let image = test_image(n, mean);
    let image_u32: Vec<u32> = image.iter().map(|&v| v as u32).collect();

    let start = Instant::now();
    let mut cpu = CpuFrc::new(n);
    println!("CPU setup: FFT plan, ring table, buffers {}", ms(start.elapsed()));
    let gpu = (!cpu_only).then(|| GpuFrc::new(n as u32));

    let pattern = [6, 19, 52, 120];
    let probe = [1, 6, 19, 30, 52, 100, 120, 500, n / 2 - 1];
    if let Some(gpu) = &gpu {
        // The FFT and rings of the GPU against the CPU's, on the same halves.
        cpu.split(&image, 1);
        gpu.upload_halves(&cpu.z);
        gpu.submit(&[FFT_2D.as_slice(), &[Stage::Rings]].concat(), true);
        let gpu_rings = gpu.read_rings();
        cpu.fft_rows();
        cpu.fft_columns();
        let cpu_rings = cpu.rings();
        let max_diff = frc_curve(&gpu_rings)
            .iter()
            .zip(frc_curve(&cpu_rings))
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f64::max);
        println!("\ncheck: same halves, GPU against CPU: largest FRC difference over all rings {max_diff:.2e}");
        let counts_equal = gpu_rings.iter().zip(&cpu_rings).all(|(g, c)| g[3] == c[3]);
        assert!(counts_equal, "the GPU and the CPU count different frequencies in a ring");
    }
    let cpu_curve = cpu.frc(&image, 2);
    let gpu_curve = gpu.as_ref().map(|gpu| gpu.frc(&image_u32, 2));
    println!("check: each with its own split (pattern rings near 1, others near 0):");
    for r in probe {
        let gpu_value = gpu_curve.as_ref().map_or(String::new(), |c| format!("GPU {:+.3}  ", c[r]));
        let mark = if pattern.contains(&r) { "  <- pattern" } else { "" };
        println!("  {r:4}: {gpu_value}CPU {:+.3}{mark}", cpu_curve[r]);
    }

    if let Some(gpu) = &gpu {
        println!("\nGPU:");
        let total = median_time(reps, || {
            gpu.frc(&image_u32, 3);
        });
        println!("  whole one_frc: upload counts, split, FFT, rings, read curve {}", ms(total));
        let resident = median_time(reps, || {
            gpu.submit(&ALL_STAGES, true);
            gpu.read_rings();
        });
        println!("  without the upload (counts already on the GPU)               {}", ms(resident));
        for (name, stages) in [
            ("split + row FFT, fused", &[Stage::SplitRowFft][..]),
            ("split alone", &[Stage::Split][..]),
            ("2D FFT", &FFT_2D[..]),
            ("  row FFT (one pass)", &[Stage::RowFft][..]),
            ("  transpose", &[Stage::Transpose][..]),
            ("rings", &[Stage::Rings][..]),
        ] {
            let t = median_time(reps, || {
                gpu.submit(stages, false);
                gpu.wait();
            });
            println!("    {name:24} {}", ms(t));
        }
        let upload = median_time(reps, || {
            gpu.upload_counts(&image_u32, 3);
            gpu.wait();
        });
        println!("    {:24} {}", "upload counts", ms(upload));
    }

    println!("\nCPU, all {} threads:", rayon::current_num_threads());
    let total = median_time(reps, || {
        cpu.frc(&image, 5);
    });
    let split_fft = median_time(reps, || cpu.split_fft_rows(&image, 4));
    let split = median_time(reps, || cpu.split(&image, 4));
    let fft_rows = median_time(reps, || cpu.fft_rows());
    let fft_columns = median_time(reps, || cpu.fft_columns());
    let column_copies = median_time(reps, || cpu.columns_through_cache(false));
    let rings = median_time(reps, || {
        cpu.rings();
    });
    println!("  whole one_frc            {}", ms(total));
    println!("    {:24} {}", "split + row FFT, fused", ms(split_fft));
    println!("    {:24} {}", "column FFT", ms(fft_columns));
    println!("    {:24} {}", "rings", ms(rings));
    println!("    {:24} {}", "(split alone)", ms(split));
    println!("    {:24} {}", "(row FFT alone)", ms(fft_rows));
    println!("    {:24} {}", "(column copies alone)", ms(column_copies));
}
