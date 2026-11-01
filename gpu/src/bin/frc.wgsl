// Fourier ring correlation of one image on the GPU, as frc.one_frc does it with method 1:
// split the counts n into k ~ binomial (n, 0.5) and n - k, Fourier transform both halves, and sum
// F1 conj(F2), |F1|^2 and |F2|^2 over rings of equal distance from the origin.
//
// Both halves go through one complex FFT, of z = k + i (n - k): their spectra are
// F1(f) = (Z(f) + conj Z(-f)) / 2 and F2(f) = (Z(f) - conj Z(-f)) / 2i. The pipeline is
// `split_fft_rows`, which splits a row of counts and transforms it in workgroup memory, then
// `transpose`, then `fft_rows` for the columns, then `rings`. The spectrum is left transposed,
// which the ring sums do not depend on. Every pass is bound by memory bandwidth, so the split
// writes nothing of its own. `split` and `fft_rows` alone check the FFT against the CPU's. The
// host replaces N and LOG2_N; N must be a power of two.

const N: u32 = __N__u;
const LOG2_N: u32 = __LOG2_N__u;
const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    len: u32,
    threads: u32,
    seed_lo: u32,
    seed_hi: u32,
}

// split
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> halves: array<vec2<f32>>;
// fft_rows
@group(0) @binding(3) var<storage, read> fft_src: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> fft_dst: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> twiddles: array<vec2<f32>>;
// transpose
@group(0) @binding(8) var<storage, read> transpose_src: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> transpose_dst: array<vec2<f32>>;
// rings
@group(0) @binding(6) var<storage, read> spectrum: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> ring_sums: array<vec4<f32>>;

// --- split: binomial (n, 0.5) by popcount, as in binom.wgsl ---

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn fmix32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x85ebca6bu;
    x ^= x >> 13u;
    x *= 0xc2b2ae35u;
    x ^= x >> 16u;
    return x;
}

var<private> s: array<u32, 4>;

fn next() -> u32 {
    let result = rotl(s[0] + s[3], 7u) + s[0];
    let t = s[1] << 9u;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11u);
    return result;
}

/// Seeds this thread's generator for one stream of the call's seed.
fn seed_generator(stream: u32) {
    var h = fmix32(params.seed_lo ^ fmix32(stream ^ fmix32(params.seed_hi)));
    for (var j = 0u; j < 4u; j++) {
        h = fmix32(h + 0x9e3779b9u);
        s[j] = h;
    }
    // xoshiro128++ must not start from all zeros.
    s[3] |= 1u;
}

/// The two halves of a count n, k ~ binomial (n, 0.5) and n - k, as a complex number.
fn split_count(n: u32) -> vec2<f32> {
    var left = n;
    var k = 0u;
    while left >= 32u {
        k += countOneBits(next());
        left -= 32u;
    }
    if left > 0u {
        k += countOneBits(next() & ((1u << left) - 1u));
    }
    return vec2<f32>(f32(k), f32(n - k));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    if thread >= params.threads {
        return;
    }
    seed_generator(thread);
    for (var i = thread; i < params.len; i += params.threads) {
        halves[i] = split_count(counts[i]);
    }
}

// --- fft_rows: FFT of one row per workgroup ---
//
// The row sits in workgroup memory in bit-reversed order, and the FFT is the usual LOG2_N
// radix-2 stages, where stage s pairs elements HALF = 2^s apart. They run four at a time: a
// thread loads the 16 elements base + t * 2^s (t < 16) of a group into registers, does stages s
// to s + 3 there, and writes them back, so the row goes through workgroup memory and a barrier
// once per four stages instead of once per stage. Stages left over when LOG2_N is not a multiple
// of four run one at a time.

var<workgroup> row: array<vec2<f32>, N>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn bit_reversed(i: u32) -> u32 {
    return reverseBits(i) >> (32u - LOG2_N);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * N;
    for (var i = lid; i < N; i += WORKGROUP_SIZE) {
        row[bit_reversed(i)] = fft_src[base + i];
    }
    workgroupBarrier();
    transform_row(lid);
    for (var i = lid; i < N; i += WORKGROUP_SIZE) {
        fft_dst[base + i] = row[i];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn split_fft_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * N;
    seed_generator(wid.x * WORKGROUP_SIZE + lid);
    for (var i = lid; i < N; i += WORKGROUP_SIZE) {
        row[bit_reversed(i)] = split_count(counts[base + i]);
    }
    workgroupBarrier();
    transform_row(lid);
    for (var i = lid; i < N; i += WORKGROUP_SIZE) {
        fft_dst[base + i] = row[i];
    }
}

/// The FFT of `row`, which holds the row in bit-reversed order; ends with a barrier.
fn transform_row(lid: u32) {
    let fused = LOG2_N - LOG2_N % 4u;
    for (var first = 0u; first < fused; first += 4u) {
        let h = 1u << first;
        for (var g = lid; g < N / 16u; g += WORKGROUP_SIZE) {
            let j = g & (h - 1u);
            let base = ((g >> first) << (first + 4u)) + j;
            var a: array<vec2<f32>, 16>;
            for (var t = 0u; t < 16u; t++) {
                a[t] = row[base + t * h];
            }
            for (var k = 0u; k < 4u; k++) {
                let stride = 1u << k;
                // exp(-2 pi i m / 2^(first + k + 1)) is entry m << shift of the table.
                let shift = LOG2_N - 1u - (first + k);
                for (var q = 0u; q < 8u; q++) {
                    let t = ((q >> k) << (k + 1u)) | (q & (stride - 1u));
                    let w = twiddles[(j + (t & (stride - 1u)) * h) << shift];
                    let v = cmul(w, a[t + stride]);
                    let u = a[t];
                    a[t] = u + v;
                    a[t + stride] = u - v;
                }
            }
            for (var t = 0u; t < 16u; t++) {
                row[base + t * h] = a[t];
            }
        }
        workgroupBarrier();
    }
    for (var stage = fused; stage < LOG2_N; stage++) {
        let half = 1u << stage;
        for (var b = lid; b < N / 2u; b += WORKGROUP_SIZE) {
            let j = b & (half - 1u);
            let i0 = ((b >> stage) << (stage + 1u)) + j;
            let i1 = i0 + half;
            // exp(-2 pi i j / (2 half)), from the table of exp(-2 pi i m / N).
            let t = cmul(twiddles[j << (LOG2_N - 1u - stage)], row[i1]);
            let u = row[i0];
            row[i0] = u + t;
            row[i1] = u - t;
        }
        workgroupBarrier();
    }
}

// --- transpose: 32 x 32 tiles through workgroup memory, so that reads and writes are both
// contiguous; the extra column keeps the tile's columns in different memory banks ---

const TILE: u32 = 32u;
var<workgroup> tile: array<vec2<f32>, 1056>;

@compute @workgroup_size(32, 8)
fn transpose(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let x0 = wid.x * TILE;
    let y0 = wid.y * TILE;
    for (var j = 0u; j < TILE; j += 8u) {
        tile[(lid.y + j) * (TILE + 1u) + lid.x] = transpose_src[(y0 + lid.y + j) * N + x0 + lid.x];
    }
    workgroupBarrier();
    for (var j = 0u; j < TILE; j += 8u) {
        transpose_dst[(x0 + lid.y + j) * N + y0 + lid.x] = tile[lid.x * (TILE + 1u) + lid.y + j];
    }
}

// --- rings: one workgroup per ring ---

// Ring r holds the frequencies f with round(|f|) = r, that is r^2 - r + 1 <= |f|^2 <= r^2 + r.
// The halves are real, so F(-f) = conj F(f), and the terms at f and -f are equal: the sums run
// over the half plane fy > 0 or (fy = 0 and fx > 0), doubled, plus the origin.

var<workgroup> partial: array<vec4<f32>, WORKGROUP_SIZE>;

fn isqrt_floor(x: i32) -> i32 {
    var r = i32(sqrt(f32(x)));
    while (r + 1) * (r + 1) <= x {
        r += 1;
    }
    while r * r > x {
        r -= 1;
    }
    return r;
}

fn at(fx: i32, fy: i32) -> vec2<f32> {
    let n = i32(N);
    return spectrum[u32((fy + n) % n) * N + u32((fx + n) % n)];
}

// F1 conj(F2), |F1|^2 and |F2|^2 at frequency (fx, fy), and a count of one.
fn ring_terms(fx: i32, fy: i32) -> vec4<f32> {
    let z = at(fx, fy);
    let zm = at(-fx, -fy);
    let c = vec2<f32>(zm.x, -zm.y);
    let f1 = 0.5 * (z + c);
    let g = z - c;
    let f2 = 0.5 * vec2<f32>(g.y, -g.x);
    return vec4<f32>(f1.x * f2.x + f1.y * f2.y, dot(f1, f1), dot(f2, f2), 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn rings(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = i32(wid.x);
    let lo = select(r * r - r + 1, 0, r == 0);
    let hi = r * r + r;
    var acc = vec4<f32>(0.0);
    for (var fy = i32(lid); fy <= r; fy += i32(WORKGROUP_SIZE)) {
        let fy2 = fy * fy;
        let a = max(lo - fy2, 0);
        var x_min = isqrt_floor(a);
        if x_min * x_min < a {
            x_min += 1;
        }
        if fy == 0 {
            x_min = max(x_min, 1);
        }
        let x_max = isqrt_floor(hi - fy2);
        for (var ax = x_min; ax <= x_max; ax++) {
            acc += ring_terms(ax, fy);
            if ax != 0 && fy != 0 {
                acc += ring_terms(-ax, fy);
            }
        }
    }
    acc *= 2.0;
    if r == 0 && lid == 0u {
        acc += ring_terms(0, 0);
    }
    partial[lid] = acc;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if lid < stride {
            partial[lid] += partial[lid + stride];
        }
        workgroupBarrier();
    }
    if lid == 0u {
        ring_sums[wid.x] = partial[0];
    }
}
