//! Compares the samplers in `rustfrc::binom_variants` with `binom_split`. Rayon uses every core
//! unless `RAYON_NUM_THREADS` says otherwise; `scripts/bench-dedibox.sh` runs this with 1 thread
//! and with all of them.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::{StdRng, ThreadRng};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustfrc::binom_variants::{
    CpuFeatures, LANES, Mwc192x, Sfc64, XoshiroLanes, binom_cdf_table, binom_popcount,
    binom_popcount_bit_buffer, binom_popcount_lanes, binom_rand_distr, fill_lanes_on,
    pois_gen_seeded, warm_cdf_tables,
};
use rustfrc::{binom_split, pois_gen, to_i32};

const SEED: u64 = 0x5EED;

type Sampler = fn(&mut [i32], u64);

/// The samplers on flat slices. `binom_split` is left out here, because it takes an ndarray.
/// PCG64, MWC192X and ChaCha12 lost to xoshiro256++ and SFC64 in every popcount and BTPE run, so
/// only their raw speed is still measured, in `rng_fill`.
fn slice_samplers() -> Vec<(&'static str, Sampler)> {
    vec![
        ("rand_distr/xoshiro", |a, s| {
            binom_rand_distr::<Xoshiro256PlusPlus>(a, s).unwrap()
        }),
        ("popcount/xoshiro/detected", |a, s| {
            binom_popcount::<Xoshiro256PlusPlus>(a, s, CpuFeatures::Detected).unwrap()
        }),
        ("popcount/xoshiro/baseline", |a, s| {
            binom_popcount::<Xoshiro256PlusPlus>(a, s, CpuFeatures::Baseline).unwrap()
        }),
        ("popcount/sfc64/detected", |a, s| {
            binom_popcount::<Sfc64>(a, s, CpuFeatures::Detected).unwrap()
        }),
        ("popcount_lanes16/avx2", |a, s| {
            binom_popcount_lanes::<16>(a, s, CpuFeatures::UpToAvx2).unwrap()
        }),
        ("popcount_lanes16/detected", |a, s| {
            binom_popcount_lanes::<16>(a, s, CpuFeatures::Detected).unwrap()
        }),
        ("popcount_lanes32/detected", |a, s| {
            binom_popcount_lanes::<32>(a, s, CpuFeatures::Detected).unwrap()
        }),
        ("bit_buffer/detected", |a, s| {
            binom_popcount_bit_buffer(a, s, CpuFeatures::Detected).unwrap()
        }),
        ("popcount_lanes16/baseline", |a, s| {
            binom_popcount_lanes::<16>(a, s, CpuFeatures::Baseline).unwrap()
        }),
        ("cdf_table/xoshiro", |a, s| {
            binom_cdf_table::<Xoshiro256PlusPlus>(a, s).unwrap()
        }),
    ]
}

/// A side x side image of Poisson (mean) counts, like the input that `binom_split` gets in FRC.
fn poisson_image(side: usize, mean: f64) -> Array2<i32> {
    let a = pois_gen(&[side, side], mean).unwrap();
    to_i32(a.view()).into_dimensionality().unwrap()
}

/// Raw generator speed on one thread: how fast each one refills 8 KiB of u64s, the buffer size
/// of `binom_popcount_lanes`. The buffer stays in L1, so stores do not bound the result.
fn rng_fill(c: &mut Criterion) {
    const WORDS: usize = 1024;
    let mut group = c.benchmark_group("rng_fill_8KiB");
    group.throughput(Throughput::Bytes((WORDS * 8) as u64));
    let mut buf = vec![0u64; WORDS];

    fn fill_with<R: Rng>(rng: &mut R, buf: &mut [u64]) {
        for w in buf.iter_mut() {
            *w = rng.next_u64();
        }
    }

    let mut thread: ThreadRng = rand::rng();
    group.bench_function("thread_rng", |b| {
        b.iter(|| fill_with(&mut thread, black_box(&mut buf)))
    });
    let mut chacha = StdRng::seed_from_u64(SEED);
    group.bench_function("chacha12", |b| {
        b.iter(|| fill_with(&mut chacha, black_box(&mut buf)))
    });
    let mut xoshiro = Xoshiro256PlusPlus::seed_from_u64(SEED);
    group.bench_function("xoshiro256pp", |b| {
        b.iter(|| fill_with(&mut xoshiro, black_box(&mut buf)))
    });
    let mut pcg = Pcg64Mcg::seed_from_u64(SEED);
    group.bench_function("pcg64mcg", |b| {
        b.iter(|| fill_with(&mut pcg, black_box(&mut buf)))
    });
    let mut sfc = Sfc64::seed_from_u64(SEED);
    group.bench_function("sfc64", |b| {
        b.iter(|| fill_with(&mut sfc, black_box(&mut buf)))
    });
    let mut mwc = Mwc192x::seed_from_u64(SEED);
    group.bench_function("mwc192x", |b| {
        b.iter(|| fill_with(&mut mwc, black_box(&mut buf)))
    });
    let mut lanes = XoshiroLanes::seed_from_u64(SEED);
    assert_eq!(WORDS % LANES, 0);
    for (name, cpu) in [
        ("xoshiro256pp_lanes/baseline", CpuFeatures::Baseline),
        ("xoshiro256pp_lanes/avx2", CpuFeatures::UpToAvx2),
        ("xoshiro256pp_lanes/detected", CpuFeatures::Detected),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| fill_lanes_on(cpu, &mut lanes, black_box(&mut buf)))
        });
    }
    group.finish();
}

/// The case from the README at a quarter of its size: a 2000 x 2000 image with mean 200.
fn binom_2000(c: &mut Criterion) {
    warm_cdf_tables();
    let image = poisson_image(2000, 200.0);
    let mut group = c.benchmark_group("binom_2000x2000_mean200");
    group.throughput(Throughput::Elements(image.len() as u64));

    group.bench_function("binom_split", |b| {
        b.iter_batched(
            || image.clone(),
            |a| binom_split(black_box(a)).unwrap(),
            BatchSize::LargeInput,
        )
    });
    for (name, sample) in slice_samplers() {
        group.bench_function(name, |b| {
            b.iter_batched(
                || image.clone(),
                |mut a| {
                    sample(black_box(a.as_slice_mut().unwrap()), SEED);
                    a
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// How each sampler scales with the counts: popcount costs n / 64 words per pixel, while the
/// table and BTPE cost about the same for any n.
fn binom_mean_sweep(c: &mut Criterion) {
    warm_cdf_tables();
    let mut group = c.benchmark_group("binom_1000x1000_by_mean");
    for mean in [5.0, 200.0, 1000.0, 5000.0] {
        let image = poisson_image(1000, mean);
        group.throughput(Throughput::Elements(image.len() as u64));
        group.bench_with_input(BenchmarkId::new("binom_split", mean), &image, |b, image| {
            b.iter_batched(
                || image.clone(),
                |a| binom_split(black_box(a)).unwrap(),
                BatchSize::LargeInput,
            )
        });
        for (name, sample) in slice_samplers() {
            group.bench_with_input(BenchmarkId::new(name, mean), &image, |b, image| {
                b.iter_batched(
                    || image.clone(),
                    |mut a| {
                        sample(black_box(a.as_slice_mut().unwrap()), SEED);
                        a
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

fn pois(c: &mut Criterion) {
    let mut group = c.benchmark_group("pois_2000x2000_mean200");
    group.throughput(Throughput::Elements(2000 * 2000));
    group.bench_function("pois_gen", |b| {
        b.iter(|| pois_gen(black_box(&[2000, 2000]), 200.0).unwrap())
    });
    group.bench_function("seeded/xoshiro", |b| {
        b.iter(|| {
            pois_gen_seeded::<Xoshiro256PlusPlus>(black_box(2000 * 2000), 200.0, SEED).unwrap()
        })
    });
    group.finish();
}

/// Short runs, so that the whole benchmark takes minutes: enough to rank samplers that differ by
/// 10% or more, which is all this stage needs.
fn quick() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = quick();
    targets = rng_fill, binom_2000, binom_mean_sweep, pois
}
criterion_main!(benches);
