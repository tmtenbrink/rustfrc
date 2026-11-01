# Changelog

## Unreleased

rustfrc 2.0 is the compiled core of frc and has no dependencies. Its functions take any array that supports the buffer protocol and return `memoryview`s instead of NumPy arrays; `numpy.asarray` wraps one without copying it.

### Added

- `one_frc(img, seed=None)` and `FrcPlan(n)`: the whole Fourier ring correlation of a square image of counts, as `frc.one_frc` 0.1 computes it with method 1. Each count is split into binomial halves, both halves are Fourier transformed and their correlation is summed over rings, in two passes over memory. Rings are binned as DIPlib's `RadialSum` bins them, so the ring counts are identical. On 8 threads of an AMD EPYC 9V45, a 4096 x 4096 image takes 18 ms with a reused plan, against 1.0 s for `frc.one_frc` 0.1.
- `two_frc(img1, img2)` and `FrcPlan.two_frc`: the FRC curve of two images.
- `ring_counts(n)`: the number of frequencies in each ring of an n x n spectrum.
- A `seed` for `binom_split` and `one_frc`. With a seed, the same input gives the same output on every CPU.
- Type stubs.

### Changed

- `binom_split` samples binomial(n, 0.5) exactly, by counting the ones among n random bits from 16 xoshiro256++ generators at once, with AVX-512 or AVX2 when the CPU has them, chosen at run time. Its sampler is about 50 times as fast as in 1.2: ~1 ms for a 2000 x 2000 array on 8 threads of an AMD EPYC 9V45.
- Every function takes any array that supports the buffer protocol, of booleans, integers or floats in native byte order, and returns a `memoryview`: int32 from `binom_split`, float64 from `one_frc` and `two_frc`, uint64 from `ring_counts`. `binom_split` and `one_frc` refuse floats, with a `TypeError`, instead of truncating them; cast them first. They refuse counts above 2**31 - 1.
- NumPy is no longer a dependency. rustfrc has none.

### Removed

- `sqr_abs` and `pois_gen`. NumPy computes both: `np.abs(a) ** 2` and `np.random.default_rng().poisson(lam, shape)`.
- Python 3.9 and 3.10 support. Both have reached their end of life. rustfrc now requires Python 3.11 or later, the first stable ABI with the buffer protocol. Wheels are tagged `cp311-abi3`.

### Security

- Development uses pytest 9.0.3 or later on every Python, which fixes its tmpdir handling (GHSA-6w46-j5rx-g56g).

## 1.2.0 - 2026-11-01

### Removed

- Python 3.8 support. rustfrc now requires Python 3.9 or later and NumPy 1.19.3 or later, because PyO3 0.30 requires Python 3.9.

### Added

- Support for Python 3.13 to 3.15 and for NumPy 2.
- Stable-ABI wheels. One wheel per platform covers CPython 3.9 and later, so a new Python version needs no new release.
- A wheel for free-threaded CPython 3.15t and later.
- An sdist, so pip can build rustfrc on platforms without a wheel. Building it needs a Rust toolchain.

### Changed

- Linux wheels need glibc 2.28 or later (manylinux_2_28), instead of 2.17.
- Internal: Updated PyO3 and rust-numpy to 0.30, ndarray to 0.17 and rand to 0.10. rustfrc uses rand directly instead of through ndarray-rand.
- Internal: Rust edition 2024.
- Internal: Development uses uv instead of Poetry, and the unused Dockerfile is gone.

### Security

- Every file on PyPI has a PEP 740 attestation, which shows the workflow that built and published it.
- Releases go through touchgate. A release publishes only after an approval with a security key, from a protected `publish` branch, with Trusted Publishing instead of an API token. The tag comes only after the release is on PyPI.
- Wheels build with a pinned Rust, maturin and manylinux image, and with a committed `Cargo.lock`. CI and the release build them with the same workflow.
- The tests run in separate jobs, on the uploaded files, so the packages they install never run next to a build that gets published.

## 1.1.5 - 2024-04-03

- **Reintroduce Python 3.8, Numpy 1.18+ support**: Turned out it was unnecessary to drop all older versions.

## 1.1.4 - 2024-04-03

- **Drop Python 3.7, 3.8 support**: Note that this is technically a breaking change, but 1.1.3 works just fine for older versions and this doesn't include much else. This is necessary to use NumPy 1.26, which is required for Python >=3.12.
- `sqr_abs` no longer performs its computation using rayon, which improves performance for arrays at least up to 4000x4000
- Added benchmarks
- `pois_gen` no longer has a hard-coded limit for the amount of elements
- Internal: Update internal Rust crate to use new PyO3 bounds API
- Internal: Update maturin to v1+, use maturin action for CI release
