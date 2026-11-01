# Changelog

## Unreleased

### Removed

- Python 3.9 support, which ended in October 2025. rustfrc now requires Python 3.10 or later and NumPy 1.21.2 or later, the oldest NumPy with wheels for Python 3.10. Wheels are tagged `cp310-abi3`.

### Security

- Development uses pytest 9.0.3 or later on every Python, which fixes its tmpdir handling (GHSA-6w46-j5rx-g56g). pytest 9 does not support Python 3.9.

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
