//! What the GPU programs in this crate share: opening a device, and reading mapped memory.

use std::time::{Duration, Instant};

/// Opens the high-performance GPU with all the limits it has, and prints what it is. A CPU
/// adapter such as llvmpipe is refused, since it measures nothing about GPUs.
pub fn open_gpu() -> (wgpu::Device, wgpu::Queue, wgpu::Limits) {
    let start = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no GPU adapter");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, {:?}, {} {})",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu, "refusing a CPU adapter");
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("no device");
    println!("setup: instance, adapter, device {}", ms(start.elapsed()));
    (device, queue, limits)
}

/// Copies GPU memory that the CPU maps uncached (write-combined) with streaming loads, which
/// read such memory a cache line at a time instead of a word at a time. wgpu's allocator maps
/// read-back buffers that way on Intel's Mesa driver, which offers cached memory only as
/// non-coherent; a plain copy of 64 MB then takes 400 ms instead of 15.
pub fn copy_from_mapped(dst: &mut [u8], src: &[u8]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") && src.as_ptr() as usize % 16 == 0 && src.len() % 64 == 0 {
        // SAFETY: the CPU has SSE4.1, and the checks above are those of the function.
        unsafe { copy_from_write_combined(dst, src) };
        return;
    }
    dst.copy_from_slice(src)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
fn copy_from_write_combined(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::{__m128i, _mm_storeu_si128, _mm_stream_load_si128};
    let s = src.as_ptr() as *const __m128i;
    let d = dst.as_mut_ptr() as *mut __m128i;
    for i in (0..src.len() / 16).step_by(4) {
        // SAFETY: i + 3 < src.len() / 16 = dst.len() / 16, and s is 16-byte aligned.
        unsafe {
            let (a, b, c, e) = (
                _mm_stream_load_si128(s.add(i)),
                _mm_stream_load_si128(s.add(i + 1)),
                _mm_stream_load_si128(s.add(i + 2)),
                _mm_stream_load_si128(s.add(i + 3)),
            );
            _mm_storeu_si128(d.add(i), a);
            _mm_storeu_si128(d.add(i + 1), b);
            _mm_storeu_si128(d.add(i + 2), c);
            _mm_storeu_si128(d.add(i + 3), e);
        }
    }
}

pub fn ms(d: Duration) -> String {
    format!("{:8.2} ms", d.as_secs_f64() * 1e3)
}

/// Runs `f` once to warm up, then `reps` times, and returns the median time.
pub fn median_time<F: FnMut()>(reps: usize, mut f: F) -> Duration {
    f();
    let mut times: Vec<Duration> = (0..reps)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[times.len() / 2]
}

/// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011; the
/// generator of Random123 and cuRAND's default): a keyed bijection of a 128-bit counter, so the
/// bits for a counter depend on nothing but the counter and the key. `frc.wgsl` computes the same.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let (mut c, mut k) = (counter, key);
    for round in 0..10 {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let p0 = M0 * c[0] as u64;
        let p1 = M1 * c[2] as u64;
        c = [(p1 >> 32) as u32 ^ c[1] ^ k[0], p1 as u32, (p0 >> 32) as u32 ^ c[3] ^ k[1], p0 as u32];
    }
    c
}

/// The split of one count n at `pixel` that `frc.wgsl` makes: k is the number of ones among the
/// first n bits of Philox4x32-10 at the counters (pixel, 0, 0, 0), (pixel, 1, 0, 0), ..., keyed by
/// the seed, so binomial (n, 0.5) exactly.
pub fn philox_split_count(n: u32, pixel: u32, seed: u64) -> u32 {
    let key = [seed as u32, (seed >> 32) as u32];
    let mut k = 0;
    let mut left = n;
    let mut block = 0;
    while left > 0 {
        for word in philox4x32_10([pixel, block, 0, 0], key) {
            let bits = left.min(32);
            k += (word & (u32::MAX >> (32 - bits))).count_ones();
            left -= bits;
            if left == 0 {
                break;
            }
        }
        block += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known answers from Random123's kat_vectors for philox4x32 with 10 rounds.
    #[test]
    fn philox_matches_random123() {
        assert_eq!(philox4x32_10([0; 4], [0; 2]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
        );
        assert_eq!(
            philox4x32_10([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0]),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
        );
    }

    #[test]
    fn philox_split_is_binomial_half() {
        for n in [0u32, 1, 31, 32, 33, 127, 128, 129, 200, 1000] {
            let samples: Vec<u32> = (0..200_000).map(|p| philox_split_count(n, p, 0xABCD ^ n as u64)).collect();
            assert!(samples.iter().all(|&k| k <= n));
            if n == 0 {
                continue;
            }
            let len = samples.len() as f64;
            let mean = samples.iter().map(|&k| k as f64).sum::<f64>() / len;
            let var = samples.iter().map(|&k| (k as f64 - mean).powi(2)).sum::<f64>() / len;
            let (m, v) = (n as f64 / 2.0, n as f64 / 4.0);
            assert!(((mean - m) / (v / len).sqrt()).abs() < 6.0, "n = {n}: mean {mean}");
            assert!(((var - v) / (2.0 * v * v / len).sqrt()).abs() < 6.0, "n = {n}: variance {var}");
        }
    }
}
