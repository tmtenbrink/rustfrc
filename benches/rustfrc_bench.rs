use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use rustfrc::{binom_split, pois_gen, sqr_abs, to_i32};
use num_complex::Complex;
use ndarray_rand::rand::prelude::{Distribution, thread_rng};
use ndarray_rand::rand_distr::Poisson;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample-10");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10);
    let size = [4000, 4000];
    let arr = pois_gen(&size, 200f64).unwrap();

    let arr_complex = arr.map(|e| {
        let mut rng = thread_rng();
        let a = Poisson::new(*e).unwrap().sample(&mut rng);
        Complex::new(*e, a.to_owned())
    });

    let arr_i32 = to_i32(arr.view());
    let pois_shape = [4000, 4000];

    group.bench_function("binom", |b| b.iter_batched(|| arr_i32.to_owned(), |a| binom_split(black_box(a)), BatchSize::SmallInput));
    group.bench_function("pois", |b| b.iter(|| pois_gen(black_box(&pois_shape), black_box(100f64))));
    group.bench_function("sqr_abs", |b| b.iter_batched(|| arr_complex.to_owned(), |a| sqr_abs(black_box(a)), BatchSize::SmallInput));
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);