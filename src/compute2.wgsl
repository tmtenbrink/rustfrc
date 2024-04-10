// This is the input
// In wgpu we create a bind group, with group index 0 and add two bindings to it
// This is binding zero
@group(0)
@binding(0)
var<storage, read_write> s: array<u32, 8>;

// This is binding one, the output
@group(0)
@binding(1)
var<storage, read_write> r: array<u32, 128>;

// This function is called 64 times (the workgroup size)
@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // var i0 = global_id.x * 2u;
    // var i1 = i0 + 1u;

    // for (var i = 0u; i < global_id.x; i++) {
    //     next();
    // }
    // var n = next();

    r[global_id.x] = global_id.x;
    //r[i1] = n[1];
}