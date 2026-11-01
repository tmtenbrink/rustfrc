// Binomial (n, 0.5) by popcount on the GPU: every count n becomes the number of ones in n random
// bits. Each thread runs its own xoshiro128++ (https://prng.di.unimi.it/xoshiro128plusplus.c),
// 32 bits at a time since WGSL has no u64, and handles the pixels thread, thread + T,
// thread + 2T, ... for T threads in all, so that neighbouring threads read neighbouring pixels.

struct Params {
    len: u32,
    threads: u32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> samples: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

// The finalizer of MurmurHash3, to spread a thread's index into its seed.
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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    if thread >= params.threads {
        return;
    }
    var h = fmix32(params.seed_lo ^ fmix32(thread ^ fmix32(params.seed_hi)));
    for (var j = 0u; j < 4u; j++) {
        h = fmix32(h + 0x9e3779b9u);
        s[j] = h;
    }
    // xoshiro128++ must not start from all zeros.
    s[3] |= 1u;

    for (var i = thread; i < params.len; i += params.threads) {
        var n = counts[i];
        var k = 0u;
        while n >= 32u {
            k += countOneBits(next());
            n -= 32u;
        }
        if n > 0u {
            k += countOneBits(next() & ((1u << n) - 1u));
        }
        samples[i] = k;
    }
}
