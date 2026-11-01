//! Times `binom_split`, `FrcPlan::one_frc` and `FrcPlan::two_frc` at several image sizes, and
//! the steps of `one_frc` at 4096 x 4096. Rayon uses every core unless `RAYON_NUM_THREADS` says
//! otherwise.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustfrc::binomial::{CpuFeatures, binom_split};
use rustfrc::frc::FrcPlan;

const SEED: u64 = 0x5EED;

/// Poisson counts with mean 100 around a few cosine waves.
fn image(n: usize) -> Vec<i32> {
    let mut image = vec![0; n * n];
    image.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(y as u64);
        for (x, v) in row.iter_mut().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * (3.0 * x as f64 + 5.0 * y as f64) / n as f64;
            *v = Poisson::new(100.0 * (1.0 + 0.4 * phase.cos()))
                .unwrap()
                .sample(&mut rng) as i32;
        }
    });
    image
}

fn split(c: &mut Criterion) {
    let mut group = c.benchmark_group("binom_split");
    for n in [1024, 4096] {
        let image = image(n);
        let mut out = image.clone();
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                out.copy_from_slice(&image);
                binom_split(black_box(&mut out), SEED, CpuFeatures::Detected).unwrap()
            })
        });
    }
    group.finish();
}

fn frc(c: &mut Criterion) {
    let mut group = c.benchmark_group("frc");
    for n in [1024, 2048, 4000, 4096, 8192] {
        let image = image(n);
        let floats: Vec<f32> = image.iter().map(|&v| v as f32).collect();
        let mut plan = FrcPlan::new(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_function(BenchmarkId::new("one_frc", n), |b| {
            b.iter(|| plan.one_frc(black_box(&image), SEED).unwrap())
        });
        group.bench_function(BenchmarkId::new("two_frc", n), |b| {
            b.iter(|| plan.two_frc(black_box(&floats), black_box(&floats)))
        });
    }
    group.finish();
}

fn steps(c: &mut Criterion) {
    let n = 4096;
    let image = image(n);
    let mut plan = FrcPlan::new(n);
    plan.one_frc(&image, SEED).unwrap();
    let mut group = c.benchmark_group("one_frc_steps_4096");
    group.bench_function("split_fft_rows", |b| {
        b.iter(|| plan.split_fft_rows(black_box(&image), SEED).unwrap())
    });
    group.bench_function("fft_columns_ring_sums", |b| {
        b.iter(|| plan.fft_columns_ring_sums())
    });
    group.finish();
}

/// Short runs, so that the whole benchmark takes minutes.
fn quick() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = quick();
    targets = split, frc, steps
}
criterion_main!(benches);
