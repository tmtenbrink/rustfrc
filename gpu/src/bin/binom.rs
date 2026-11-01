//! Times binomial (n, 0.5) sampling of an image of Poisson counts on the GPU (wgpu, `binom.wgsl`)
//! and on the CPU (rustfrc), and checks the GPU's samples against the binomial distribution.
//!
//!   cargo run --release --bin binom -- [SIDE] [MEAN] [REPS]      defaults: 4000 200 10
//!
//! Set WGPU_BACKEND or WGPU_ADAPTER_NAME to pick another device; a CPU adapter such as llvmpipe
//! is refused, since it measures nothing about GPUs.

use std::time::Instant;

use rustfrc_gpu::{copy_from_mapped, median_time, ms, open_gpu};

use bytemuck::{Pod, Zeroable};
use rand_xoshiro::Xoshiro256PlusPlus;
use rustfrc::binom_variants::{CpuFeatures, binom_popcount, binom_popcount_lanes};
use rustfrc::{binom_split, pois_gen, to_i32};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    threads: u32,
    seed_lo: u32,
    seed_hi: u32,
}

/// The device, the pipeline and buffers sized for one image, kept between calls.
struct GpuSampler {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    counts: wgpu::Buffer,
    samples: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    len: usize,
}

impl GpuSampler {
    fn new(len: usize) -> Self {
        let (device, queue, limits) = open_gpu();
        let start = Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binom.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("binom.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let size = (len * 4) as u64;
        assert!(size <= limits.max_storage_buffer_binding_size, "image too large for one binding");
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let counts = buffer("counts", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let samples = buffer("samples", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = buffer("readback", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: samples.as_entire_binding() },
            ],
        });
        let sampler = Self { device, queue, pipeline, params, counts, samples, readback, bind_group, len };
        sampler.wait();
        println!("setup: shader, pipeline, buffers {}", ms(start.elapsed()));
        sampler
    }

    fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    fn set_params(&self, threads: u32, seed: u64) {
        let params = Params { len: self.len as u32, threads, seed_lo: seed as u32, seed_hi: (seed >> 32) as u32 };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    fn upload(&self, counts: &[u32]) {
        self.queue.write_buffer(&self.counts, 0, bytemuck::cast_slice(counts));
    }

    /// Records `dispatches` runs of the kernel, and the copy of the samples for reading back.
    fn submit(&self, threads: u32, dispatches: usize, read_back: bool) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(threads.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        if read_back {
            encoder.copy_buffer_to_buffer(&self.samples, 0, &self.readback, 0, None);
        }
        self.queue.submit([encoder.finish()]);
    }

    fn read_back(&self, out: &mut [u32]) {
        self.readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        self.wait();
        let mapped = self.readback.get_mapped_range(..).unwrap();
        copy_from_mapped(bytemuck::cast_slice_mut(out), &mapped);
        drop(mapped);
        self.readback.unmap();
    }

    /// Upload, sample and read back: what a call from Python would cost.
    fn sample(&self, counts: &[u32], out: &mut [u32], threads: u32, seed: u64) {
        self.set_params(threads, seed);
        self.upload(counts);
        self.submit(threads, 1, true);
        self.read_back(out);
    }
}

/// Mean and variance of the samples against n / 2 and n / 4, as z-scores.
fn check_constant(gpu: &GpuSampler, n: u32, threads: u32) {
    let counts = vec![n; gpu.len];
    let mut out = vec![0; gpu.len];
    gpu.sample(&counts, &mut out, threads, 0xC0FFEE ^ n as u64);
    let len = out.len() as f64;
    let mean = out.iter().map(|&k| k as f64).sum::<f64>() / len;
    let var = out.iter().map(|&k| (k as f64 - mean).powi(2)).sum::<f64>() / len;
    let max = *out.iter().max().unwrap();
    if n == 0 {
        assert_eq!(max, 0);
        println!("check n = {n:5}: all zero");
        return;
    }
    let (m, v) = (n as f64 / 2.0, n as f64 / 4.0);
    let z_mean = (mean - m) / (v / len).sqrt();
    // The variance of a sample variance is about 2 v^2 / len for counts that are close to normal.
    let z_var = (var - v) / (2.0 * v * v / len).sqrt();
    println!("check n = {n:5}: mean {mean:9.3} (z {z_mean:+.2}), variance {var:9.3} (z {z_var:+.2}), max {max}");
    assert!(max <= n && z_mean.abs() < 6.0 && z_var.abs() < 6.0, "GPU samples are off for n = {n}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args.get(1).map_or(4000, |s| s.parse().unwrap());
    let mean: f64 = args.get(2).map_or(200.0, |s| s.parse().unwrap());
    let reps: usize = args.get(3).map_or(10, |s| s.parse().unwrap());

    let image = to_i32(pois_gen(&[side, side], mean).unwrap().view());
    let image_i32: Vec<i32> = image.iter().copied().collect();
    let image_u32: Vec<u32> = image_i32.iter().map(|&n| n as u32).collect();
    let len = image_u32.len();
    println!("image: {side} x {side}, Poisson mean {mean}, {reps} reps, median times\n");

    let gpu = GpuSampler::new(len);
    let threads_options = [1u32 << 14, 1 << 16, 1 << 18, 1 << 20];

    println!();
    for n in [0, 1, 7, 31, 32, 33, 200, 1000] {
        check_constant(&gpu, n, 1 << 16);
    }

    println!("\nGPU, kernel only (counts resident, 10 dispatches per submit):");
    gpu.upload(&image_u32);
    gpu.wait();
    for threads in threads_options {
        gpu.set_params(threads, 1);
        let t = median_time(reps, || {
            gpu.submit(threads, 10, false);
            gpu.wait();
        }) / 10;
        println!("  {threads:8} threads {}", ms(t));
    }

    println!("\nGPU, upload + kernel + read back:");
    let mut out = vec![0u32; len];
    for threads in threads_options {
        let t = median_time(reps, || gpu.sample(&image_u32, &mut out, threads, 2));
        println!("  {threads:8} threads {}", ms(t));
    }
    let upload = median_time(reps, || {
        gpu.upload(&image_u32);
        gpu.wait();
    });
    let copy = median_time(reps, || {
        gpu.submit(1 << 16, 0, true);
        gpu.wait();
    });
    let map = median_time(reps, || {
        gpu.readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        gpu.wait();
        gpu.readback.unmap();
    });
    let read_plain = median_time(reps.min(3), || {
        gpu.readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        gpu.wait();
        out.copy_from_slice(bytemuck::cast_slice(&gpu.readback.get_mapped_range(..).unwrap()));
        gpu.readback.unmap();
    });
    let read = median_time(reps, || gpu.read_back(&mut out));
    println!(
        "  of which upload {}, GPU copy to the readback buffer {}, map {}, map + CPU read {} \
         (with plain loads {})",
        ms(upload),
        ms(copy),
        ms(map),
        ms(read),
        ms(read_plain)
    );
    let host = vec![1u32; len];
    let memcpy = median_time(reps, || out.copy_from_slice(&host));
    println!("  for scale, a plain CPU memcpy of the same 64 MB {}", ms(memcpy));

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("\nCPU, all {cores} threads:");
    let t = median_time(reps, || {
        binom_split(image.clone()).unwrap();
    });
    println!("  binom_split (with a clone)      {}", ms(t));
    let mut buf = image_i32.clone();
    let t = median_time(reps, || {
        buf.copy_from_slice(&image_i32);
        binom_popcount::<Xoshiro256PlusPlus>(&mut buf, 3, CpuFeatures::Detected).unwrap();
    });
    println!("  popcount/xoshiro/detected       {}", ms(t));
    let t = median_time(reps, || {
        buf.copy_from_slice(&image_i32);
        binom_popcount_lanes::<8>(&mut buf, 3, CpuFeatures::Detected).unwrap();
    });
    println!("  popcount_lanes/detected         {}", ms(t));
}
