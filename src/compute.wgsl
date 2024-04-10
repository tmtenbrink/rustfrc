// https://github.com/cupy/cupy/pull/1356
// THE GENERATION OF BINOMIAL RANDOM VARIATES WOLFGANG HORMANN
// https://link.springer.com/article/10.1007/s00453-015-0077-8#Bib1

fn add(a0: u32, a1: u32, b0: u32, b1: u32) -> array<u32, 2> {
    var c0 = a0 + b0;
    var c1 = a1 + b1;

    if (c0 < a0) {
        c1 = c1 + 1u;
    }

    return array<u32, 2>(c0, c1);
}

// k must non-negative and <64
fn lshift(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var b0: u32 = 0u;
    var b1: u32 = a0 << k;
    if (k <= 32u) {
        b0 = b1;
        b1 = (a1 << k) + (a0 >> k);
    }
    return array<u32, 2>(b0, b1);
}

// k must be less than 64
fn rshift(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var b0: u32;
    var b1: u32 = 0u;
    if (k > 32u) {
        b0 = a1 >> (k - 32u);
    } else {
        b0 = (a0 >> k) + (a1 << (32u - k));
        b1 = a1 >> k;
    }
    return array<u32, 2>(b0, b1);
}

fn rotl(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var l = lshift(a0, a1, k);
    var r = rshift(a0, a1, 64u - k);
    var b0: u32 = l[0] | r[0];
    var b1: u32 = l[1] | r[1];

    return array<u32, 2>(b0, b1);
}

@group(0)
@binding(0)
var<storage, read_write> s: array<u32, 8>;

@group(0)
@binding(1)
var<storage, read_write> r: array<u32, 128>;

fn next() -> array<u32, 2> {
    // 0-1 = 0
    // 2-3 = 1
    // 4-5 = 2
    // 6-7 = 3
    var a03 = add(s[0], s[1], s[6], s[7]);
    var r1 = rotl(a03[0], a03[1], 23u);
    var result = r1;
    //var result = add(r1[0], r1[1], s[0], s[1]);

	// var result = rotl(s[0] + s[3], 23) + s[0];

	let t = lshift(s[2], s[3], 17u);

	// const uint64_t t = s[1] << 17;

    s[4] = s[4] ^ s[0];
    s[5] = s[5] ^ s[1];

    s[6] = s[6] ^ s[2];
    s[7] = s[7] ^ s[3];

    s[2] = s[2] ^ s[4];
    s[3] = s[3] ^ s[5];

    s[0] = s[0] ^ s[6];
    s[1] = s[1] ^ s[7];


    //	s[2] ^= s[0];
    //	s[3] ^= s[1];
    //	s[1] ^= s[2];
    //	s[0] ^= s[3];

    s[4] = s[4] ^ t[0];
    s[5] = s[5] ^ t[1];

	// s[2] ^= t;

    var r2 = rotl(s[6], s[7], 45u);
    s[6] = r2[0];
    s[7] = r2[1];


	//s[3] = rotl(s[3], 45);

	return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

//void jump(void) {
//	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
//
//	uint64_t s0 = 0;
//	uint64_t s1 = 0;
//	uint64_t s2 = 0;
//	uint64_t s3 = 0;
//	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
//		for(int b = 0; b < 64; b++) {
//			if (JUMP[i] & UINT64_C(1) << b) {
//				s0 ^= s[0];
//				s1 ^= s[1];
//				s2 ^= s[2];
//				s3 ^= s[3];
//			}
//			next();
//		}
//
//	s[0] = s0;
//	s[1] = s1;
//	s[2] = s2;
//	s[3] = s3;
//}
//
//
//
///* This is the long-jump function for the generator. It is equivalent to
//   2^192 calls to next(); it can be used to generate 2^64 starting points,
//   from each of which jump() will generate 2^64 non-overlapping
//   subsequences for parallel distributed computations. */
//
//void long_jump(void) {
//	static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
//
//	uint64_t s0 = 0;
//	uint64_t s1 = 0;
//	uint64_t s2 = 0;
//	uint64_t s3 = 0;
//	for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
//		for(int b = 0; b < 64; b++) {
//			if (LONG_JUMP[i] & UINT64_C(1) << b) {
//				s0 ^= s[0];
//				s1 ^= s[1];
//				s2 ^= s[2];
//				s3 ^= s[3];
//			}
//			next();
//		}
//
//	s[0] = s0;
//	s[1] = s1;
//	s[2] = s2;
//	s[3] = s3;
//}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < global_id.x; i++) {
        next();
    }
    var n = next();

    r[i0] = n[0];
    r[i1] = n[1];
}