//! Times `FrcPlan::one_frc` at several image sizes, and its steps at 4096 x 4096. Rayon uses every
//! core unless `RAYON_NUM_THREADS` says otherwise.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustfrc::frc::{FrcPlan, pattern_image};

const SEED: u64 = 0x5EED;

fn one_frc(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_frc");
    for n in [1024, 2048, 4000, 4096, 8192] {
        let image = pattern_image(n, 100.0, 1);
        let mut plan = FrcPlan::new(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &image, |b, image| {
            b.iter(|| plan.one_frc(black_box(image), SEED).unwrap())
        });
    }
    group.finish();
}

fn steps(c: &mut Criterion) {
    let n = 4096;
    let image = pattern_image(n, 100.0, 1);
    let mut plan = FrcPlan::new(n);
    plan.one_frc(&image, SEED).unwrap();
    let mut group = c.benchmark_group("one_frc_steps_4096");
    group.bench_function("split_fft_rows", |b| b.iter(|| plan.split_fft_rows(black_box(&image), SEED).unwrap()));
    group.bench_function("fft_columns", |b| b.iter(|| plan.fft_columns()));
    group.bench_function("ring_sums", |b| b.iter(|| plan.ring_sums()));
    group.bench_function("split", |b| b.iter(|| plan.split(black_box(&image), SEED).unwrap()));
    group.bench_function("fft_rows", |b| b.iter(|| plan.fft_rows()));
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
    targets = one_frc, steps
}
criterion_main!(benches);
