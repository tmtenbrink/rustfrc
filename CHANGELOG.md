# Changelog

## 1.1.5 2024-04-03

* **Reintroduce Python 3.8, Numpy 1.18+ support**: Turned out it was unnecessary to drop all older versions.

## 1.1.4 2024-04-03

* **Drop Python 3.7, 3.8 support**: Note that this is technically a breaking change, but 1.1.3 works just fine for older versions and this doesn't include much else. This is necessary to use NumPy 1.26, which is required for Python >=3.12.
* `sqr_abs` no longer performs its computation using rayon, which improves performance for arrays at least up to 4000x4000
* Added benchmarks
* `pois_gen` no longer has a hard-coded limit for the amount of elements

### Internal

* Update internal Rust crate to use new PyO3 bounds API
* Update maturin to v1+, use maturin action for CI release