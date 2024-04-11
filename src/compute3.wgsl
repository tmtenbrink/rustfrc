// Based on https://prng.di.unimi.it/xoshiro256plusplus.c

// performs a << k, where a is a 64 bit number represented by two u32's
fn lshift(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var b0: u32 = 0u;
    var b1: u32 = 0u;
    if (k >= 64) {
        // In this case everything is shifted to zero, so we do nothing
    } else if (k >= 32) {
        // In this case the leading part disappears, the trailing part now becomes the leading part
        // But shifted by the remaining shift subtracted by 32
        b0 = a1 << (k - 32);
    } else {
        // We just shift the leading part by k
        // For the a1 part, we need to get the bits that 'make it across'
        b0 = (a0 << k) + (a1 >> (32 - k));
        // The part that doesn't make it just needs to be shifted
        b1 = a1 << k;
    }
    return array<u32, 2>(b0, b1);
}

// performs a >> k, where a is a 64 bit number represented by two u32's
// k must be strictly less than 64
fn rshift(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var b0: u32 = 0u;
    var b1: u32 = 0u;
    if (k >= 32u) {
        // In this case the trailing part disappears, the leading part becomes trailing part
        // But shifted by the shift subtracted by 32
        b1 = a0 >> (k - 32);
    } else {
        b0 = a0 >> k;
        b1 = (a1 >> k) + (a0 << (32 - k));
    }
    return array<u32, 2>(b0, b1);
}

fn add(a0: u32, a1: u32, b0: u32, b1: u32) -> array<u32, 2> {
    var c0 = a0 + b0;
    var c1 = a1 + b1;

    if (c1 < a1) {
        c0 = c0 + 1u;
    }

    return array<u32, 2>(c0, c1);
}

fn rotl(a0: u32, a1: u32, k: u32) -> array<u32, 2> {
    var l = lshift(a0, a1, k);
    var r = rshift(a0, a1, 64u - k);
    var b0: u32 = l[0] | r[0];
    var b1: u32 = l[1] | r[1];

    // == (x << k) | (x >> (64 - k));

    return array<u32, 2>(b0, b1);
}

fn next_result(s: array<u32, 8>) -> array<u32, 2> {
    // 0-1 = 0
    // 2-3 = 1
    // 4-5 = 2
    // 6-7 = 3
    let a03 = add(s[0], s[1], s[6], s[7]);
    let r1 = rotl(a03[0], a03[1], 23u);
    return add(r1[0], r1[1], s[0], s[1]);

    // == const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
}

fn next(s_in: array<u32, 8>) -> array<u32, 8> {
	var s = s_in;
    let t = lshift(s[2], s[3], 17u);

	// == const uint64_t t = s[1] << 17;

    s[4] = s[4] ^ s[0];
    s[5] = s[5] ^ s[1];

    s[6] = s[6] ^ s[2];
    s[7] = s[7] ^ s[3];

    s[2] = s[2] ^ s[4];
    s[3] = s[3] ^ s[5];

    s[0] = s[0] ^ s[6];
    s[1] = s[1] ^ s[7];


    // == s[2] ^= s[0];
    // == s[3] ^= s[1];
    // == s[1] ^= s[2];
    // == s[0] ^= s[3];

    s[4] = s[4] ^ t[0];
    s[5] = s[5] ^ t[1];

	// == s[2] ^= t;

    var r2 = rotl(s[6], s[7], 45u);
    s[6] = r2[0];
    s[7] = r2[1];

	// == s[3] = rotl(s[3], 45);

	return s;
}

fn jump(s_in: array<u32, 8>) -> array<u32, 8> {
    var jump_arr: array<u32, 8> = array(1023216314, 403621587, 4039719212, 3584430694, 3762276778, 2841126424, 699491868, 967564357);

    return jump_loop(s_in, jump_arr);
}

fn long_jump(s_in: array<u32, 8>) -> array<u32, 8> {
    var jump_arr: array<u32, 8> = array(4278045631, 1994480958, 475148211, 3305131588, 2236539457, 2003894377, 718005813, 957389744);

    return jump_loop(s_in, jump_arr);
}


fn jump_loop(s_in: array<u32, 8>, jump_in: array<u32, 8>) -> array<u32, 8> {
    var jump_arr = jump_in;
    var s = s_in;
    var s0 = 0u;
	var s1 = 0u;
	var s2 = 0u;
	var s3 = 0u;
    var s4 = 0u;
    var s5 = 0u;
	var s6 = 0u;
	var s7 = 0u;
	for(var i = 0u; i < 8u; i++) {
        for (var b = 0u; b < 32u; b++) {
            if ((jump_arr[i] & (1u << b)) != 0) {
                s0 = s0 ^ s[0];
                s1 = s1 ^ s[1];
                s2 = s2 ^ s[2];
                s3 = s3 ^ s[3];
                s4 = s4 ^ s[4];
                s5 = s5 ^ s[5];
                s6 = s6 ^ s[6];
                s7 = s7 ^ s[7];
            }
            s = next(s);
        }
    }

	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
    s[4] = s4;
    s[5] = s5;
    s[6] = s6;
    s[7] = s7;

    return s;
}

@group(0)
@binding(0)
var<storage> seed: array<u32, 8>;

@group(0)
@binding(1)
var<storage, read_write> r: array<u32>;



@compute
@workgroup_size(64)
fn main_lshift_5(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 4u; i++) {
        var res = lshift(seed[i0], seed[i1], 5u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_lshift_33(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 4u; i++) {
        var res = lshift(seed[i0], seed[i1], 33u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_rshift_7(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 4u; i++) {
        var res = rshift(seed[i0], seed[i1], 7u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_rshift_35(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 4u; i++) {
        var res = rshift(seed[i0], seed[i1], 35u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 4u;
    var i1 = i0 + 1u;
    var i2 = i0 + 2u;
    var i3 = i0 + 3u;

    for (var i = 0u; i < 2u; i++) {
        var res = add(seed[i0], seed[i1], seed[i2], seed[i3]);
        r[global_id.x * 2u] = res[0];
        r[global_id.x * 2u + 1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_rotl_23(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 2u; i++) {
        var res = rotl(seed[i0], seed[i1], 23u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

@compute
@workgroup_size(64)
fn main_rotl_45(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i0 = global_id.x * 2u;
    var i1 = i0 + 1u;

    for (var i = 0u; i < 2u; i++) {
        var res = rotl(seed[i0], seed[i1], 45u);
        r[i0] = res[0];
        r[i1] = res[1];
    }
}

// Workgroup size can be no more than 256

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) wgid : vec3<u32>) {
    var state: array<u32, 8> = seed;
    
    for (var i = 0u; i < wgid.y; i++) {
        state = long_jump(state);
    }

    for (var i = 0u; i < local_id.x; i++) {
        state = jump(state);
    }

    for (var i = 0u; i < 512u; i++) {
        var i0 = (global_id.x*512 + i) * 2u;
        var i1 = i0 + 1u;
        
        state = next(state);
        var n = next_result(state);

        r[i0] = n[0];
        r[i1] = n[1];
    }
    
}