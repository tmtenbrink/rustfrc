//! Experimental samplers for binomial (n, 0.5), to compare against `binom_split` in the
//! benchmarks. Each one takes a flat slice of counts, replaces every count n with a sample of
//! binomial (n, 0.5), and fails if a count is negative. Unlike `binom_split`, they take a seed:
//! every chunk of `CHUNK_LEN` pixels gets its own generator seeded from it, so the output depends
//! on the seed alone and not on how rayon schedules the chunks.

use std::convert::Infallible;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use rand::{Rng, SeedableRng, TryRng};
use rand_distr::{Binomial, Distribution, Poisson, PoissonError};
use rand_xoshiro::SplitMix64;
use rayon::prelude::*;

use crate::ToUsizeError;

/// Number of pixels that one generator samples, and so the unit of parallel work.
pub const CHUNK_LEN: usize = 1 << 14;

fn chunk_seed(seed: u64, chunk_index: usize) -> u64 {
    seed ^ (chunk_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Runs `sample_chunk` on every chunk in parallel, with a generator of type `R` for each chunk.
/// A chunk that holds a negative count is left unchanged, and the call then fails.
fn sample_chunks_seeded<R, F>(a: &mut [i32], seed: u64, sample_chunk: F) -> Result<(), ToUsizeError>
where
    R: SeedableRng,
    F: Fn(&mut R, &mut [i32]) + Sync,
{
    let negative = AtomicBool::new(false);
    a.par_chunks_mut(CHUNK_LEN)
        .enumerate()
        .for_each(|(i, chunk)| {
            if chunk.iter().any(|&n| n < 0) {
                negative.store(true, Ordering::Relaxed);
                return;
            }
            let mut rng = R::seed_from_u64(chunk_seed(seed, i));
            sample_chunk(&mut rng, chunk);
        });
    if negative.into_inner() {
        Err(ToUsizeError {})
    } else {
        Ok(())
    }
}

/// The current method (`rand_distr`'s BTPE) with one seeded generator per chunk, instead of
/// fetching the thread-local generator for every pixel.
pub fn binom_rand_distr<R: SeedableRng + Rng>(
    a: &mut [i32],
    seed: u64,
) -> Result<(), ToUsizeError> {
    sample_chunks_seeded::<R, _>(a, seed, |rng, chunk| {
        for x in chunk {
            *x = Binomial::new(*x as u64, 0.5).unwrap().sample(rng) as i32;
        }
    })
}

/// Counts the ones among the low `bits` bits of `word`, where `1 <= bits <= 64`.
#[inline(always)]
fn ones_in_low_bits(word: u64, bits: u32) -> u32 {
    (word & (u64::MAX >> (64 - bits))).count_ones()
}

/// Exact binomial (n, 0.5): the number of ones in n random bits, drawn 64 at a time.
#[inline(always)]
fn popcount_binomial<R: Rng>(rng: &mut R, n: u32) -> u32 {
    let mut k = 0;
    let mut left = n;
    while left >= 64 {
        k += rng.next_u64().count_ones();
        left -= 64;
    }
    if left > 0 {
        k += ones_in_low_bits(rng.next_u64(), left);
    }
    k
}

/// Which instructions a sampler may use beyond the x86-64 baseline that wheels are built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuFeatures {
    /// Only the baseline (SSE2), whatever the CPU has.
    Baseline,
    /// What the CPU reports at run time, up to AVX2 and `popcnt`: never AVX-512.
    UpToAvx2,
    /// Everything the CPU reports at run time that a sampler has a path for, AVX-512 included.
    Detected,
}

/// The code that `binom_popcount_lanes` runs, as `CpuFeatures` and the CPU decide it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanesPath {
    Portable,
    /// Rotates as two shifts and an or, popcount as a table lookup (`vpshufb`, `vpsadbw`).
    Avx2,
    /// Native rotate (`vprolq`) and popcount (`vpopcntq`).
    Avx512,
}

impl CpuFeatures {
    fn popcnt(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self != Self::Baseline && is_x86_feature_detected!("popcnt")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn avx2_popcnt(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.popcnt() && is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn avx512_popcnt(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self == Self::Detected
                && self.avx2_popcnt()
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vpopcntdq")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// The path `binom_popcount_lanes` takes on this CPU.
    pub fn lanes_path(self) -> LanesPath {
        if self.avx512_popcnt() {
            LanesPath::Avx512
        } else if self.avx2_popcnt() {
            LanesPath::Avx2
        } else {
            LanesPath::Portable
        }
    }
}

#[inline(always)]
fn popcount_chunk<R: Rng>(rng: &mut R, chunk: &mut [i32]) {
    for x in chunk {
        *x = popcount_binomial(rng, *x as u32) as i32;
    }
}

/// `popcount_chunk` compiled with `popcnt`, so that `count_ones` is one instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
fn popcount_chunk_popcnt<R: Rng>(rng: &mut R, chunk: &mut [i32]) {
    popcount_chunk(rng, chunk)
}

/// Popcount sampling with a scalar generator `R`.
pub fn binom_popcount<R: SeedableRng + Rng>(
    a: &mut [i32],
    seed: u64,
    cpu: CpuFeatures,
) -> Result<(), ToUsizeError> {
    let popcnt = cpu.popcnt();
    sample_chunks_seeded::<R, _>(a, seed, |rng, chunk| {
        #[cfg(target_arch = "x86_64")]
        if popcnt {
            // SAFETY: `CpuFeatures::popcnt` checked that the CPU has `popcnt`.
            unsafe { popcount_chunk_popcnt(rng, chunk) };
            return;
        }
        popcount_chunk(rng, chunk)
    })
}

/// SFC64 (Chris Doty-Humphrey, PractRand 0.96, shifts 25, 11 and 3), one of the fastest scalar
/// generators in the PRNG Shootout (https://prng.di.unimi.it/).
#[derive(Clone)]
pub struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl SeedableRng for Sfc64 {
    type Seed = [u8; 24];

    fn from_seed(seed: Self::Seed) -> Self {
        let word = |i: usize| u64::from_le_bytes(seed[8 * i..8 * i + 8].try_into().unwrap());
        let mut rng = Self {
            a: word(0),
            b: word(1),
            c: word(2),
            counter: 1,
        };
        // PractRand discards the first outputs, which still show the seed.
        for _ in 0..12 {
            rng.next_u64();
        }
        rng
    }
}

impl TryRng for Sfc64 {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.next_u64() >> 32) as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(25).wrapping_add(tmp);
        Ok(tmp)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        fill_bytes_by_words(self, dst);
        Ok(())
    }
}

/// MWC192X: multiply-with-carry with the X scrambler, from Vigna, "Scrambled Multiply-with-Carry
/// Pseudorandom Number Generators" (arXiv:2610.27313), one of the fastest in the PRNG Shootout.
#[derive(Clone)]
pub struct Mwc192x {
    x: u64,
    y: u64,
    c: u64,
}

const MWC192X_A2: u64 = 0xff4e_24dd_418f_5433;

impl SeedableRng for Mwc192x {
    type Seed = [u8; 16];

    fn from_seed(seed: Self::Seed) -> Self {
        let word = |i: usize| u64::from_le_bytes(seed[8 * i..8 * i + 8].try_into().unwrap());
        // A carry of 1 keeps the state away from the fixed points of the recurrence.
        Self {
            x: word(0),
            y: word(1),
            c: 1,
        }
    }
}

impl TryRng for Mwc192x {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.next_u64() >> 32) as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let result = self.y ^ self.c;
        let t = MWC192X_A2 as u128 * self.x as u128 + self.c as u128;
        self.x = self.y;
        self.y = t as u64;
        self.c = (t >> 64) as u64;
        Ok(result)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        fill_bytes_by_words(self, dst);
        Ok(())
    }
}

fn fill_bytes_by_words<R: Rng>(rng: &mut R, dst: &mut [u8]) {
    for chunk in dst.chunks_mut(8) {
        let word = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
}

/// Default number of independent xoshiro256++ streams that `XoshiroLanes` steps together. The
/// PRNG Shootout's vectorized xoshiro256++ runs fastest with 16 under Clang, whose backend rustc
/// shares; with AVX2, 16 lanes of state fill all 16 vector registers.
pub const LANES: usize = 16;

/// `L` xoshiro256++ generators stored lane by lane, so that one step of all of them compiles to
/// vector instructions: `L / 4` AVX2 registers per state word.
#[derive(Clone)]
pub struct XoshiroLanes<const L: usize = LANES> {
    s: [[u64; L]; 4],
}

impl<const L: usize> XoshiroLanes<L> {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::seed_from_u64(seed);
        let mut s = [[0; L]; 4];
        for word in s.iter_mut().flatten() {
            *word = sm.next_u64();
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

    /// Fills `out`, whose length must be a multiple of `L`.
    #[inline(always)]
    pub fn fill(&mut self, out: &mut [u64]) {
        let (blocks, rest) = out.as_chunks_mut::<L>();
        assert!(
            rest.is_empty(),
            "length {} is not a multiple of {L}",
            out.len()
        );
        for block in blocks {
            self.next_block(block);
        }
    }
}

/// Words that `binom_popcount_bit_buffer` generates at a time, per chunk.
const BIT_BUFFER_WORDS: usize = 1024;

/// Random words from `XoshiroLanes`, handed out in order.
struct BitBuffer {
    rng: XoshiroLanes,
    words: [u64; BIT_BUFFER_WORDS],
    pos: usize,
}

impl BitBuffer {
    #[inline(always)]
    fn new(rng: XoshiroLanes) -> Self {
        Self {
            rng,
            words: [0; BIT_BUFFER_WORDS],
            pos: BIT_BUFFER_WORDS,
        }
    }

    /// Exact binomial (n, 0.5), with the n bits taken from the buffer.
    #[inline(always)]
    fn binomial(&mut self, n: u32) -> u32 {
        let mut k = 0;
        let mut left = n as usize;
        while left > 0 {
            if self.pos == BIT_BUFFER_WORDS {
                self.rng.fill(&mut self.words);
                self.pos = 0;
            }
            let needed = left.div_ceil(64);
            let take = needed.min(BIT_BUFFER_WORDS - self.pos);
            let words = &self.words[self.pos..self.pos + take];
            self.pos += take;
            if take < needed {
                k += words.iter().map(|w| w.count_ones()).sum::<u32>();
                left -= 64 * take;
            } else {
                let (last, full) = words.split_last().unwrap();
                k += full.iter().map(|w| w.count_ones()).sum::<u32>();
                k += ones_in_low_bits(*last, (left - 64 * (take - 1)) as u32);
                left = 0;
            }
        }
        k
    }
}

#[inline(always)]
fn bit_buffer_chunk(rng: XoshiroLanes, chunk: &mut [i32]) {
    let mut bits = BitBuffer::new(rng);
    for x in chunk {
        *x = bits.binomial(*x as u32) as i32;
    }
}

/// `bit_buffer_chunk` compiled for AVX2 and `popcnt`. The x86-64 baseline that wheels are
/// built for has neither, so without this `count_ones` is a sequence of shifts and masks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
fn bit_buffer_chunk_avx2(rng: XoshiroLanes, chunk: &mut [i32]) {
    bit_buffer_chunk(rng, chunk)
}

/// Exact binomial (n, 0.5) for `L` pixels at once, one per lane: lane i takes the bits of
/// pixel i from generator lane i, 64 at a time, until the largest count of the group is
/// covered. Every step is the same for all lanes, so it compiles to vector code: the mask is a
/// variable shift (AVX2's `vpsllvq`) and the popcount a table lookup (`vpshufb`, `vpsadbw`),
/// since AVX2 has no 64-bit popcount instruction.
#[inline(always)]
fn popcount_binomial_lanes<const L: usize>(rng: &mut XoshiroLanes<L>, n: &[u32; L]) -> [u32; L] {
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

#[inline(always)]
fn popcount_lanes_chunk<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    let (groups, tail) = chunk.as_chunks_mut::<L>();
    for group in groups {
        let k = popcount_binomial_lanes(rng, &group.map(|x| x as u32));
        *group = k.map(|k| k as i32);
    }
    if !tail.is_empty() {
        let mut n = [0; L];
        for (n, x) in n.iter_mut().zip(tail.iter()) {
            *n = *x as u32;
        }
        let k = popcount_binomial_lanes(rng, &n);
        for (x, k) in tail.iter_mut().zip(k) {
            *x = k as i32;
        }
    }
}

/// `popcount_lanes_chunk` compiled for AVX2 (and `popcnt`, should LLVM keep any of it scalar).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
fn popcount_lanes_chunk_avx2<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    popcount_lanes_chunk(rng, chunk)
}

/// `popcount_lanes_chunk` compiled for AVX-512, where `rotate_left` becomes `vprolq` and
/// `count_ones` becomes `vpopcntq` (AVX512_VPOPCNTDQ: Ice Lake and later, Zen 4 and later).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512vpopcntdq,avx2,popcnt")]
fn popcount_lanes_chunk_avx512<const L: usize>(rng: &mut XoshiroLanes<L>, chunk: &mut [i32]) {
    popcount_lanes_chunk(rng, chunk)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn fill_lanes_avx2(rng: &mut XoshiroLanes, out: &mut [u64]) {
    rng.fill(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2")]
fn fill_lanes_avx512(rng: &mut XoshiroLanes, out: &mut [u64]) {
    rng.fill(out)
}

/// `XoshiroLanes::fill` on the path that `cpu` and the CPU allow, for the generator benchmark.
pub fn fill_lanes_on(cpu: CpuFeatures, rng: &mut XoshiroLanes, out: &mut [u64]) {
    match cpu.lanes_path() {
        LanesPath::Portable => rng.fill(out),
        // SAFETY: `CpuFeatures::lanes_path` checked that the CPU has the features of each path.
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx2 => unsafe { fill_lanes_avx2(rng, out) },
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx512 => unsafe { fill_lanes_avx512(rng, out) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("only x86-64 has vector paths"),
    }
}

/// Popcount sampling with `XoshiroLanes` that generates words for the lanes together but hands
/// them out to one pixel at a time, using AVX2 and `popcnt` if `cpu` allows them. Superseded by
/// `binom_popcount_lanes`; kept to compare the two.
pub fn binom_popcount_bit_buffer(
    a: &mut [i32],
    seed: u64,
    cpu: CpuFeatures,
) -> Result<(), ToUsizeError> {
    let avx2 = cpu.avx2_popcnt();
    sample_chunks_seeded::<SplitMix64, _>(a, seed, |sm, chunk| {
        let rng = XoshiroLanes::seed_from_u64(sm.next_u64());
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: `CpuFeatures::avx2_popcnt` checked that the CPU has both.
            unsafe { bit_buffer_chunk_avx2(rng, chunk) };
            return;
        }
        bit_buffer_chunk(rng, chunk)
    })
}

/// Popcount sampling with one pixel per lane of an `L`-lane `XoshiroLanes` (see
/// `popcount_binomial_lanes`), on the path that `cpu` and the CPU allow (`CpuFeatures::lanes_path`).
pub fn binom_popcount_lanes<const L: usize>(
    a: &mut [i32],
    seed: u64,
    cpu: CpuFeatures,
) -> Result<(), ToUsizeError> {
    let path = cpu.lanes_path();
    sample_chunks_seeded::<SplitMix64, _>(a, seed, |sm, chunk| {
        let mut rng = XoshiroLanes::<L>::seed_from_u64(sm.next_u64());
        sample_lanes_chunk(path, &mut rng, chunk)
    })
}

/// `popcount_lanes_chunk` on `path`.
fn sample_lanes_chunk<const L: usize>(
    path: LanesPath,
    rng: &mut XoshiroLanes<L>,
    chunk: &mut [i32],
) {
    match path {
        LanesPath::Portable => popcount_lanes_chunk(rng, chunk),
        // SAFETY: `CpuFeatures::lanes_path` checked that the CPU has the features of each path.
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx2 => unsafe { popcount_lanes_chunk_avx2(rng, chunk) },
        #[cfg(target_arch = "x86_64")]
        LanesPath::Avx512 => unsafe { popcount_lanes_chunk_avx512(rng, chunk) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("only x86-64 has vector paths"),
    }
}

/// Counts that `LanesSplitter::split_chunk` samples at a time, on the stack.
const SPLIT_BLOCK: usize = 4096;

/// `binom_popcount_lanes` driven one chunk at a time by the caller, so that the split can be
/// fused with other work on the same memory, such as an FFT of each row while it is in cache.
/// Chunk `index` of a seed gets the generator of chunk `index` of `binom_popcount_lanes`, so
/// with chunks of `CHUNK_LEN` counts both give the same samples.
pub struct LanesSplitter<const L: usize> {
    seed: u64,
    path: LanesPath,
}

impl<const L: usize> LanesSplitter<L> {
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
    ) -> Result<(), ToUsizeError> {
        assert_eq!(counts.len(), out.len());
        if counts.iter().any(|&n| n < 0) {
            return Err(ToUsizeError {});
        }
        let mut sm = SplitMix64::seed_from_u64(chunk_seed(self.seed, index));
        let mut rng = XoshiroLanes::<L>::seed_from_u64(sm.next_u64());
        let mut k = [0i32; SPLIT_BLOCK];
        for (c, o) in counts.chunks(SPLIT_BLOCK).zip(out.chunks_mut(SPLIT_BLOCK)) {
            let k = &mut k[..c.len()];
            k.copy_from_slice(c);
            sample_lanes_chunk(self.path, &mut rng, k);
            for ((o, &n), &k) in o.iter_mut().zip(c).zip(k.iter()) {
                *o = f(n, k);
            }
        }
        Ok(())
    }
}

/// Largest n that `binom_cdf_table` keeps a table for; larger counts fall back to BTPE.
pub const CDF_TABLE_MAX_N: usize = 1024;

/// For every n up to `CDF_TABLE_MAX_N`, the cumulative distribution of binomial (n, 0.5) scaled
/// to the range of a u64: entry k is P(X <= k) * 2^64, for k = 0..n (P(X <= n) = 1 is implied).
struct CdfTables {
    starts: Vec<usize>,
    thresholds: Vec<u64>,
}

impl CdfTables {
    fn build() -> Self {
        let mut starts = Vec::with_capacity(CDF_TABLE_MAX_N + 2);
        let mut thresholds = Vec::new();
        for n in 0..=CDF_TABLE_MAX_N {
            starts.push(thresholds.len());
            let ln_half_n = n as f64 * 0.5f64.ln();
            let mut ln_choose = 0.0f64;
            let mut cdf = 0.0f64;
            for k in 0..n {
                cdf += (ln_choose + ln_half_n).exp();
                // `as` saturates, so a sum that rounds to 1 becomes u64::MAX.
                thresholds.push((cdf * 18_446_744_073_709_551_616.0) as u64);
                ln_choose += ((n - k) as f64 / (k + 1) as f64).ln();
            }
        }
        starts.push(thresholds.len());
        Self { starts, thresholds }
    }

    fn get() -> &'static Self {
        static TABLES: OnceLock<CdfTables> = OnceLock::new();
        TABLES.get_or_init(Self::build)
    }

    #[inline(always)]
    fn sample(&self, n: usize, u: u64) -> u32 {
        let cdf = &self.thresholds[self.starts[n]..self.starts[n + 1]];
        cdf.partition_point(|&t| t <= u) as u32
    }
}

/// Inversion: one random u64 per pixel, looked up in a table of the cumulative distribution.
/// Probabilities are exact to f64 precision; counts above `CDF_TABLE_MAX_N` use BTPE.
pub fn binom_cdf_table<R: SeedableRng + Rng>(a: &mut [i32], seed: u64) -> Result<(), ToUsizeError> {
    let tables = CdfTables::get();
    sample_chunks_seeded::<R, _>(a, seed, |rng, chunk| {
        for x in chunk {
            let n = *x as usize;
            *x = if n <= CDF_TABLE_MAX_N {
                tables.sample(n, rng.next_u64()) as i32
            } else {
                Binomial::new(n as u64, 0.5).unwrap().sample(rng) as i32
            };
        }
    })
}

/// Builds the tables of `binom_cdf_table` now, so that a benchmark does not time it.
pub fn warm_cdf_tables() {
    CdfTables::get();
}

/// `pois_gen` with the distribution built once and one seeded generator per chunk.
pub fn pois_gen_seeded<R: SeedableRng + Rng>(
    len: usize,
    lambda: f64,
    seed: u64,
) -> Result<Vec<f64>, PoissonError> {
    let poisson = Poisson::new(lambda)?;
    let mut a = vec![0.0; len];
    a.par_chunks_mut(CHUNK_LEN)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut rng = R::seed_from_u64(chunk_seed(seed, i));
            for x in chunk {
                *x = poisson.sample(&mut rng);
            }
        });
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    type Sampler = fn(&mut [i32], u64) -> Result<(), ToUsizeError>;

    fn samplers() -> Vec<(&'static str, Sampler)> {
        vec![
            ("rand_distr/chacha12", binom_rand_distr::<StdRng>),
            ("rand_distr/xoshiro", binom_rand_distr::<Xoshiro256PlusPlus>),
            ("popcount/xoshiro", |a, s| {
                binom_popcount::<Xoshiro256PlusPlus>(a, s, CpuFeatures::Detected)
            }),
            ("popcount/xoshiro/baseline", |a, s| {
                binom_popcount::<Xoshiro256PlusPlus>(a, s, CpuFeatures::Baseline)
            }),
            ("popcount/sfc64", |a, s| {
                binom_popcount::<Sfc64>(a, s, CpuFeatures::Detected)
            }),
            ("popcount/mwc192x", |a, s| {
                binom_popcount::<Mwc192x>(a, s, CpuFeatures::Detected)
            }),
            ("popcount_lanes32", |a, s| {
                binom_popcount_lanes::<32>(a, s, CpuFeatures::Detected)
            }),
            ("popcount_lanes16", |a, s| {
                binom_popcount_lanes::<16>(a, s, CpuFeatures::Detected)
            }),
            ("popcount_lanes8/baseline", |a, s| {
                binom_popcount_lanes::<8>(a, s, CpuFeatures::Baseline)
            }),
            ("bit_buffer", |a, s| {
                binom_popcount_bit_buffer(a, s, CpuFeatures::Detected)
            }),
            ("cdf_table/xoshiro", binom_cdf_table::<Xoshiro256PlusPlus>),
        ]
    }

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

    #[test]
    fn samplers_follow_binomial_half() {
        for (name, sample) in samplers() {
            for n in [
                0u64, 1, 2, 7, 63, 64, 65, 127, 200, 1000, 1024, 1025, 5000, 40_000,
            ] {
                let mut a = vec![n as i32; 100_000];
                sample(&mut a, 0xF00D ^ n).unwrap();
                assert_binomial_half(name, n, &a);
            }
        }
    }

    #[test]
    fn samplers_reject_negative_counts() {
        for (name, sample) in samplers() {
            let mut a = vec![5; 3 * CHUNK_LEN];
            a[CHUNK_LEN + 7] = -1;
            assert!(sample(&mut a, 1).is_err(), "{name}");
        }
    }

    #[test]
    fn samplers_depend_only_on_seed() {
        for (name, sample) in samplers() {
            let input: Vec<i32> = (0..5 * CHUNK_LEN as i32).map(|i| i % 300).collect();
            let (mut a, mut b) = (input.clone(), input);
            sample(&mut a, 42).unwrap();
            sample(&mut b, 42).unwrap();
            assert_eq!(a, b, "{name}");
        }
    }

    #[test]
    fn splitter_matches_binom_popcount_lanes() {
        let counts: Vec<i32> = (0..5 * CHUNK_LEN as i32 + 77).map(|i| i % 700).collect();
        let mut a = counts.clone();
        binom_popcount_lanes::<16>(&mut a, 9, CpuFeatures::Detected).unwrap();
        let splitter = LanesSplitter::<16>::new(9, CpuFeatures::Detected);
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

    #[test]
    fn detected_paths_match_baseline_paths() {
        let input: Vec<i32> = (0..3 * CHUNK_LEN as i32).map(|i| i % 2000).collect();
        let (mut a, mut b) = (input.clone(), input.clone());
        let mut c = input.clone();
        binom_popcount_lanes::<16>(&mut a, 7, CpuFeatures::Detected).unwrap();
        binom_popcount_lanes::<16>(&mut b, 7, CpuFeatures::Baseline).unwrap();
        binom_popcount_lanes::<16>(&mut c, 7, CpuFeatures::UpToAvx2).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
        let (mut a, mut b) = (input.clone(), input);
        binom_popcount::<Xoshiro256PlusPlus>(&mut a, 7, CpuFeatures::Detected).unwrap();
        binom_popcount::<Xoshiro256PlusPlus>(&mut b, 7, CpuFeatures::Baseline).unwrap();
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod reference_outputs {
    use super::*;

    /// Outputs of the C code in the PRNG Shootout (cpu/other/sfc64-speed.c and
    /// cpu/mwc/MWC192X-speed.c), started from a = 1, b = 2, c = 3 and x = 5, y = 7, c = 1.
    #[test]
    fn match_c_reference() {
        let mut sfc = Sfc64 {
            a: 1,
            b: 2,
            c: 3,
            counter: 1,
        };
        let mut mwc = Mwc192x { x: 5, y: 7, c: 1 };
        let got: Vec<(u64, u64)> = (0..3).map(|_| (sfc.next_u64(), mwc.next_u64())).collect();
        assert_eq!(
            got,
            [
                (4, 6),
                (31, 18196434007968490756),
                (905969730, 18096309981672066415)
            ]
        );
    }
}
