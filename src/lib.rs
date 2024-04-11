use std::borrow::Cow;
use std::time::Instant;
use ndarray_rand::rand::{Rng, thread_rng};
use ndarray_rand::rand::rngs::OsRng;
use wgpu::util::DeviceExt;

fn u32_from_64 (u: u64) -> [u32; 2] {
    let leading_part = u >> 32;
    let trailing_part = (u ^ (leading_part << 32)) as u32; 
    [leading_part as u32, trailing_part]
}

fn u64_from_2_32 (a: &[u32; 2]) -> u64 {
    let trailing_part = a[1] as u64;
    let leading_part = (a[0] as u64) << 32;
    trailing_part + leading_part
}

fn vec_u32_from_64 (u_vec: Vec<u64>) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(u_vec.len()*2);
    for u in u_vec {
        let u32_arr = u32_from_64(u);
        out.push(u32_arr[0]);
        out.push(u32_arr[1]);
    }
    out
}

fn vec_u64_from_2_32 (u_vec: Vec<u32>) -> Vec<u64> {
    if u_vec.len() % 2 != 0 {
        panic!("Length must be multiple of 2!")
    }
    let mut out: Vec<u64> = Vec::with_capacity(u_vec.len()/2);
    for u in u_vec.chunks_exact(2) {
        out.push(u64_from_2_32(u.try_into().unwrap()))
    }
    out
}


pub async fn run() {
    let now = Instant::now();
    
    
    println!("{} pre", now.elapsed().as_millis());
    let mut os_rng = OsRng::default();
    // let mut rng = thread_rng();
    // let mut thr_vec: Vec<u64> = Vec::with_capacity(4096*1000);
    // for _i in 0..1024 {
    //     let mut arr2 = [0u64; 4096];
    //     rng.fill(&mut arr2); 
    //     thr_vec.extend_from_slice(&arr2)
    // }

    let seed: [u32; 8] = os_rng.gen();
    println!("{} genned", now.elapsed().as_millis());
    // let nonzero_gens = thr_vec.iter().filter(|u| **u != 0).count();
    // println!("nonzero genned: {} ", nonzero_gens);
    println!("{} pre-exec", now.elapsed().as_millis());
    let gpu_add = execute_gpu("main", &seed).await.unwrap();
    println!("{} did exec", now.elapsed().as_millis());
    let gpu_add_64 = vec_u64_from_2_32(gpu_add.clone());
    println!("{} to u64", now.elapsed().as_millis());
    //println!("Seed: [{:?}]", seed_to_64);
    println!("New: [{:?}]", gpu_add_64.iter().take(30).collect::<Vec<&u64>>());
    let nonzeros = gpu_add_64.iter().filter(|u| **u != 0).count();
    println!("Nonzeros: [{:?}]", nonzeros);
    println!("Length: [{:?}]", gpu_add_64.len());
    println!("{} fin", now.elapsed().as_millis());
    //println!("New Rust: [{:?}]", seed_add);
    
}

async fn execute_gpu(entry: &'static str, seed: &[u32; 8]) -> Option<Vec<u32>> {
    let now = Instant::now();
    println!("eg {}", now.elapsed().as_millis());
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    println!("eg {} adapt", now.elapsed().as_millis());

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

        println!("eg {} device", now.elapsed().as_millis());

    let info = adapter.get_info();
    // skip this on LavaPipe temporarily
    if info.vendor == 0x10005 {
        return None;
    }

    // 1600 seems sweet spot
    // After 6500+ it doesn't seem to work anymore
    // The total latency of starting and connecting the device is ~120+ ms
    // Then around 10 ms for the actual compu in the case of just 1 dispatched workgroup
    execute_gpu_inner(entry, &device, &queue, seed, 128).await
}

const WORKGROUP_SIZE: u32 = 256;
const INVOKE_SIZE: u32 = 512;
// The buffer limit is around 256 MB

async fn execute_gpu_inner(
    entry: &'static str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    seed: &[u32; 8],
    dispatch_size: u32
) -> Option<Vec<u32>> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute3.wgsl"))),
    });
    let now = Instant::now();
    println!("ig {} mod", now.elapsed().as_millis());

    let zero: Vec<u32> = vec![0; (dispatch_size*WORKGROUP_SIZE*INVOKE_SIZE*2).try_into().unwrap()];
    //let v_big_zero = v_big.as_slice();

    // Gets the size in bytes of the buffer.
    let slice_size = zero.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_buffer_out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&zero),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(seed),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
    });

    println!("ig {} buf", now.elapsed().as_millis());

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: entry,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer_out.as_entire_binding(),
            }
        ],
    });

    println!("ig {} bind", now.elapsed().as_millis());

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute seeds");
        cpass.dispatch_workgroups(dispatch_size, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }

    println!("ig {} dispatched", now.elapsed().as_millis());
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer_out, 0, &staging_buffer, 0, size);

    println!("ig {} copy", now.elapsed().as_millis());

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    println!("ig {} map", now.elapsed().as_millis());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    println!("ig {} resolved", now.elapsed().as_millis());

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        println!("ig {} cast", now.elapsed().as_millis());

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory

        println!("ig {} dropped", now.elapsed().as_millis());

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn rotl(x: u64, k: u64) -> u64 {
    (x << k) | (x >> (64 - k))
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[pollster::test]
    async fn test_lshift_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u32; 8] = os_rng.gen();
        let seed_to_64 = vec_u64_from_2_32(Vec::from(seed.to_owned()));
        let seed_shift: Vec<u64> = seed_to_64.iter().map(|u| u << 5).collect();
        let gpu_shift = execute_gpu("main_lshift_5", &seed).await.unwrap();
        let gpu_shift = vec_u64_from_2_32(gpu_shift);
        seed_shift.iter().zip(gpu_shift.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_lshift_33_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u32; 8] = os_rng.gen();
        let seed_to_64 = vec_u64_from_2_32(Vec::from(seed.to_owned()));
        let seed_shift: Vec<u64> = seed_to_64.iter().map(|u| u << 33).collect();
        let gpu_shift = execute_gpu("main_lshift_33", &seed).await.unwrap();
        let gpu_shift = vec_u64_from_2_32(gpu_shift);
        seed_shift.iter().zip(gpu_shift.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_rshift_7_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u32; 8] = os_rng.gen();
        let seed_to_64 = vec_u64_from_2_32(Vec::from(seed.to_owned()));
        let seed_shift: Vec<u64> = seed_to_64.iter().map(|u| u >> 7).collect();
        let gpu_shift = execute_gpu("main_rshift_7", &seed).await.unwrap();
        let gpu_shift = vec_u64_from_2_32(gpu_shift);
        seed_shift.iter().zip(gpu_shift.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_rshift_35_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u32; 8] = os_rng.gen();
        let seed_to_64 = vec_u64_from_2_32(Vec::from(seed.to_owned()));
        let seed_shift: Vec<u64> = seed_to_64.iter().map(|u| u >> 35).collect();
        let gpu_shift = execute_gpu("main_rshift_35", &seed).await.unwrap();
        let gpu_shift = vec_u64_from_2_32(gpu_shift);
        seed_shift.iter().zip(gpu_shift.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_add_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u64; 4] = os_rng.gen();
        let seed_to_64: Vec<u64> = seed.iter().map(|u| u/2).collect();
        let seed = vec_u32_from_64(seed_to_64.to_owned());
        let seed_add: Vec<u64> = seed_to_64.chunks_exact(2).map(|u_arr| u_arr[0] + u_arr[1]).collect();
        let gpu_add = execute_gpu("main_add", &seed.clone().try_into().unwrap()).await.unwrap();
        let gpu_add_64 = vec_u64_from_2_32(gpu_add.clone());
        seed_add.iter().zip(gpu_add_64.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_rotl_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u64; 4] = os_rng.gen();
        let seed_to_64: Vec<u64> = seed.iter().map(|u| u/2).collect();
        let seed = vec_u32_from_64(seed_to_64.to_owned());
        let seed_rotl: Vec<u64> = seed_to_64.iter().map(|u| rotl(*u, 23u64)).collect();
        let gpu_rotl = execute_gpu("main_rotl_23", &seed.clone().try_into().unwrap()).await.unwrap();
        let gpu_rotl_64 = vec_u64_from_2_32(gpu_rotl.clone());
        seed_rotl.iter().zip(gpu_rotl_64.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }

    #[pollster::test]
    async fn test_rotl_45_eq() {
        let mut os_rng = OsRng::default();
        let seed: [u64; 4] = os_rng.gen();
        let seed_to_64: Vec<u64> = seed.iter().map(|u| u/2).collect();
        let seed = vec_u32_from_64(seed_to_64.to_owned());
        let seed_rotl: Vec<u64> = seed_to_64.iter().map(|u| rotl(*u, 45u64)).collect();
        let gpu_rotl = execute_gpu("main_rotl_45", &seed.clone().try_into().unwrap()).await.unwrap();
        let gpu_rotl_64 = vec_u64_from_2_32(gpu_rotl.clone());
        seed_rotl.iter().zip(gpu_rotl_64.iter()).for_each(|(r, g)| {
            assert_eq!(r, g)
        });
    }
}