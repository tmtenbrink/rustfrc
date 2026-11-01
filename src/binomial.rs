//! Exact binomial (n, 0.5) sampling: the number of ones in n random bits. Sixteen xoshiro256++
//! generators run side by side, one pixel per lane, so that every step compiles to vector code,
//! with AVX-512 or AVX2 when the CPU has them. Every path gives the same samples.
//!
//! Every chunk of `CHUNK_LEN` counts gets its own generator, seeded from the call's seed and the
//! chunk's index, so the samples depend on the seed alone and not on how rayon schedules the
//! chunks.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};

use rand::SeedableRng;
use rand_xoshiro::SplitMix64;
use rayon::prelude::*;

/// Number of counts that one generator samples, and so the unit of parallel work.
pub const CHUNK_LEN: usize = 1 << 14;

/// Number of xoshiro256++ streams that `XoshiroLanes` steps together. 16 lanes give 16
/// independent dependency chains, which is what keeps the vector units busy; 8 lanes are slower.
pub const LANES: usize = 16;

/// A count that binomial (n, 0.5) is not defined for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegativeCount;

impl Display for NegativeCount {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("counts must not be negative")
    }
}

impl std::error::Error for NegativeCount {}

fn chunk_seed(seed: u64, chunk_index: usize) -> u64 {
    seed ^ (chunk_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Which instructions the sampler may use beyond the x86-64 baseline that wheels are built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFeatures {
    /// Only the baseline (SSE2), whatever the CPU has.
    Baseline,
    /// What the CPU reports at run time, up to AVX2 and `popcnt`: never AVX-512.
    UpToAvx2,
    /// Everything the CPU reports at run time that the sampler has a path for.
    Detected,
}

/// The code that the sampler runs, as `CpuFeatures` and the CPU decide it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanesPath {
    Portable,
    /// Rotates as two shifts and an or, popcount as a table lookup (`vpshufb`, `vpsadbw`).
    Avx2,
    /// Native rotate (`vprolq`) and popcount (`vpopcntq`).
    Avx512,
}

impl CpuFeatures {
    /// The path the sampler takes on this CPU.
    pub fn lanes_path(self) -> LanesPath {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = self != Self::Baseline
                && is_x86_feature_detected!("popcnt")
                && is_x86_feature_detected!("avx2");
            let avx512 = self == Self::Detected
                && avx2
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vpopcntdq");
            if avx512 {
                return LanesPath::Avx512;
            }
            if avx2 {
                return LanesPath::Avx2;
            }
        }
        LanesPath::Portable
    }
}

/// `L` xoshiro256++ generators stored lane by lane, so that one step of all of them compiles to
/// vector instructions.
#[derive(Clone)]
pub struct XoshiroLanes<const L: usize = LANES> {
    s: [[u64; L]; 4],
}

impl<const L: usize> XoshiroLanes<L> {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::seed_from_u64(seed);
        let mut s = [[0; L]; 4];
        for word in s.iter_mut().flatten() {
            *word = rand::Rng::next_u64(&mut sm);
        }
        Self { s }
    }

    /// Steps every lane once and writes their outputs to `out`.
    #[inline(always)]
    pub fn next_block(&mut self, out: &mut [u64; L]) {
        let [s0, s1, s2, s3] = &mut self.s;
        for i in 0..L {
            out[i] = s0[i]
                .wrapping_add(s3[i])
                .rotate_left(23)
                .wrapping_add(s0[i]);
            let t = s1[i] << 17;
            s2[i] ^= s0[i];
            s3[i] ^= s1[i];
            s1[i] ^= s2[i];
            s0[i] ^= s3[i];
            s2[i] ^= t;
            s3[i] = s3[i].rotate_left(45);
        }
    }
}

/// Exact binomial (n, 0.5) for `L` counts at once, one per lane: lane i takes the bits of count
/// i from generator lane i, 64 at a time, until the largest count of the group is covered. Every
/// step is the same for all lanes, so it compiles to vector code: the mask is a variable shift
/// (`vpsllvq`) and the popcount a table lookup on AVX2, which has no 64-bit popcount.
#[inline(always)]
fn binomial_lanes<const L: usize>(rng: &mut XoshiroLanes<L>, n: &[u32; L]) -> [u32; L] {
    let max = n.iter().copied().max().unwrap_or(0);
    let mut k = [0u64; L];
    let mut words = [0u64; L];
    let mut covered = 0u32;
    while covered < max {
        rng.next_block(&mut words);
        for i in 0..L {
            let left = n[i].saturating_sub(covered);
            let mask = if left >= 64 {
                u64::MAX
            } else {
                (1u64 << left) - 1
            };
            k[i] += (words[i] & mask).count_ones() as u64;
        }
        covered += 64;
    }
    k.map(|k| k as u32)
}

/// Replaces every count of `chunk`, which must not be negative, with a sample.
#[inline(always)]
fn sample_chunk_portable<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    let (groups, tail) = chunk.as_chunks_mut::<L>();
    for group in groups {
        let k = binomial_lanes(rng, &group.map(|x| x as u32));
        *group = k.map(|k| k as i32);
    }
    if !tail.is_empty() {
        let mut n = [0; L];
        for (n, x) in n.iter_mut().zip(tail.iter()) {
            *n = *x as u32;
        }
        let k = binomial_lanes(rng, &n);
        for (x, k) in tail.iter_mut().zip(k) {
            *x = k as i32;
        }
    }
}

/// `sample_chunk_portable` compiled for AVX2 (and `popcnt`, should LLVM keep any of it scalar).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
fn sample_chunk_avx2<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    sample_chunk_portable(rng, chunk)
}

/// `sample_chunk_portable` compiled for AVX-512, where `rotate_left` becomes `vprolq` and
/// `count_ones` becomes `vpopcntq` (AVX512_VPOPCNTDQ: Ice Lake and later, Zen 4 and later).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512vpopcntdq,avx2,popcnt")]
fn sample_chunk_avx512<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    sample_chunk_portable(rng, chunk)
}

fn sample_chunk<const L: usize>(path: LanesPath, rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    match path {
        LanesPath::Portable => sample_chunk_portable(rng, chunk),
        // SAFETY: `CpuFeatures::lanes_path` checked that the CPU has the features of each path.
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx2 => unsafe { sample_chunk_avx2(rng, chunk) },
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx512 => unsafe { sample_chunk_avx512(rng, chunk) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("only x86-64 has vector paths"),
    }
}

/// The generator of chunk `index` for `seed`.
fn chunk_generator(seed: u64, index: usize) -> XoshiroLanes {
    let mut sm = SplitMix64::seed_from_u64(chunk_seed(seed, index));
    XoshiroLanes::seed_from_u64(rand::Rng::next_u64(&mut sm))
}

/// Replaces every count n of `counts` with a sample of binomial (n, 0.5), in parallel, using the
/// path that `cpu` and the CPU allow. Fails, and leaves the chunks with a negative count as they
/// are, if a count is negative.
pub fn binom_split(counts: &mut [i32], seed: u64, cpu: CpuFeatures) -> Result<(), NegativeCount> {
    let path = cpu.lanes_path();
    let negative = AtomicBool::new(false);
    counts
        .par_chunks_mut(CHUNK_LEN)
        .enumerate()
        .for_each(|(i, chunk)| {
            if chunk.iter().any(|&n| n < 0) {
                negative.store(true, Ordering::Relaxed);
                return;
            }
            sample_chunk(path, &mut chunk_generator(seed, i), chunk);
        });
    if negative.into_inner() {
        Err(NegativeCount)
    } else {
        Ok(())
    }
}

/// Counts that `Splitter::split_chunk` samples at a time, on the stack.
const SPLIT_BLOCK: usize = 4096;

/// `binom_split` driven one chunk at a time by the caller, so that the split can be fused with
/// other work on the same memory, such as an FFT of each row while it is in cache. Chunk `index`
/// gets the generator of chunk `index` of `binom_split`, so with chunks of `CHUNK_LEN` counts
/// both give the same samples.
pub struct Splitter {
    seed: u64,
    path: LanesPath,
}

impl Splitter {
    pub fn new(seed: u64, cpu: CpuFeatures) -> Self {
        Self {
            seed,
            path: cpu.lanes_path(),
        }
    }

    /// Samples k ~ binomial (n, 0.5) for every count n of `counts`, chunk `index` of the image,
    /// and writes `f(n, k)` to the same place in `out`; `counts` stays as it is.
    pub fn split_chunk<T>(
        &self,
        index: usize,
        counts: &[i32],
        out: &mut [T],
        f: impl Fn(i32, i32) -> T,
    ) -> Result<(), NegativeCount> {
        assert_eq!(counts.len(), out.len());
        if counts.iter().any(|&n| n < 0) {
            return Err(NegativeCount);
        }
        let mut rng = chunk_generator(self.seed, index);
        let mut k = [0i32; SPLIT_BLOCK];
        for (c, o) in counts.chunks(SPLIT_BLOCK).zip(out.chunks_mut(SPLIT_BLOCK)) {
            let k = &mut k[..c.len()];
            k.copy_from_slice(c);
            sample_chunk(self.path, &mut rng, k);
            for ((o, &n), &k) in o.iter_mut().zip(c).zip(k.iter()) {
                *o = f(n, k);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn ln_choose(n: u64, k: u64) -> f64 {
        (0..k).map(|i| ((n - i) as f64 / (i + 1) as f64).ln()).sum()
    }

    /// Pearson's chi-squared test against binomial (n, 0.5), with bins merged until each expects
    /// at least 20 samples. The bound sits about six standard deviations above the mean of the
    /// statistic, and the seeds are fixed, so the test does not flake.
    fn assert_binomial_half(name: &str, n: u64, samples: &[i32]) {
        let total = samples.len() as f64;
        let mut counts = vec![0u64; n as usize + 1];
        for &k in samples {
            assert!((0..=n as i32).contains(&k), "{name}: n = {n} gave {k}");
            counts[k as usize] += 1;
        }
        let (mut chi2, mut bins) = (0.0, 0);
        let (mut expected, mut observed) = (0.0, 0.0);
        for k in 0..=n {
            expected += (ln_choose(n, k) + n as f64 * 0.5f64.ln()).exp() * total;
            observed += counts[k as usize] as f64;
            if expected >= 20.0 || k == n {
                chi2 += (observed - expected).powi(2) / expected.max(1e-300);
                bins += 1;
                (expected, observed) = (0.0, 0.0);
            }
        }
        let df = (bins - 1).max(1) as f64;
        let bound = df + 6.0 * (2.0 * df).sqrt() + 10.0;
        assert!(
            chi2 < bound,
            "{name}: n = {n}, chi2 = {chi2:.1} over {bins} bins, bound {bound:.1}"
        );
    }

    const PATHS: [CpuFeatures; 3] = [
        CpuFeatures::Baseline,
        CpuFeatures::UpToAvx2,
        CpuFeatures::Detected,
    ];

    #[test]
    fn samples_follow_binomial_half() {
        for cpu in PATHS {
            for n in [
                0u64, 1, 2, 7, 63, 64, 65, 127, 200, 1000, 1024, 1025, 5000, 40_000,
            ] {
                let mut a = vec![n as i32; 100_000];
                binom_split(&mut a, 0xF00D ^ n, cpu).unwrap();
                assert_binomial_half(&format!("{cpu:?}"), n, &a);
            }
        }
    }

    #[test]
    fn negative_counts_are_refused() {
        let mut a = vec![5; 3 * CHUNK_LEN];
        a[CHUNK_LEN + 7] = -1;
        assert_eq!(
            binom_split(&mut a, 1, CpuFeatures::Detected),
            Err(NegativeCount)
        );
    }

    #[test]
    fn samples_depend_only_on_the_seed() {
        let input: Vec<i32> = (0..5 * CHUNK_LEN as i32).map(|i| i % 300).collect();
        let (mut a, mut b, mut c) = (input.clone(), input.clone(), input);
        binom_split(&mut a, 42, CpuFeatures::Detected).unwrap();
        binom_split(&mut b, 42, CpuFeatures::Detected).unwrap();
        binom_split(&mut c, 43, CpuFeatures::Detected).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn every_path_gives_the_same_samples() {
        let input: Vec<i32> = (0..3 * CHUNK_LEN as i32).map(|i| i % 2000).collect();
        let samples: Vec<Vec<i32>> = PATHS
            .iter()
            .map(|&cpu| {
                let mut a = input.clone();
                binom_split(&mut a, 7, cpu).unwrap();
                a
            })
            .collect();
        assert_eq!(samples[0], samples[1]);
        assert_eq!(samples[0], samples[2]);
    }

    #[test]
    fn splitter_matches_binom_split() {
        let counts: Vec<i32> = (0..5 * CHUNK_LEN as i32 + 77).map(|i| i % 700).collect();
        let mut a = counts.clone();
        binom_split(&mut a, 9, CpuFeatures::Detected).unwrap();
        let splitter = Splitter::new(9, CpuFeatures::Detected);
        let mut b = vec![(0, 0); counts.len()];
        for (i, (c, o)) in counts
            .chunks(CHUNK_LEN)
            .zip(b.chunks_mut(CHUNK_LEN))
            .enumerate()
        {
            splitter.split_chunk(i, c, o, |n, k| (n, k)).unwrap();
        }
        assert!(
            b.iter()
                .zip(&a)
                .zip(&counts)
                .all(|((&(n, k), &ka), &c)| n == c && k == ka)
        );
        let mut negative = counts[..100].to_vec();
        negative[50] = -3;
        assert!(
            splitter
                .split_chunk(0, &negative, &mut b[..100], |n, k| (n, k))
                .is_err()
        );
    }

    /// Each lane is a plain xoshiro256++ generator.
    #[test]
    fn lanes_are_xoshiro256plusplus() {
        let mut lanes = XoshiroLanes::<4>::seed_from_u64(5);
        let mut scalar: Vec<Xoshiro256PlusPlus> = (0..4)
            .map(|i| {
                let mut seed = [0u8; 32];
                for (w, bytes) in seed.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    bytes.copy_from_slice(&lanes.s[w][i].to_le_bytes());
                }
                Xoshiro256PlusPlus::from_seed(seed)
            })
            .collect();
        let mut out = [0; 4];
        for _ in 0..10 {
            lanes.next_block(&mut out);
            for (o, s) in out.iter().zip(scalar.iter_mut()) {
                assert_eq!(*o, s.next_u64());
            }
        }
    }
}
