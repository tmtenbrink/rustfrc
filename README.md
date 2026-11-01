# rustfrc

![GitHub release](https://flat.badgen.net/github/release/tiptenbrink/rustfrc)
![License](https://flat.badgen.net/github/license/tiptenbrink/rustfrc)

rustfrc is the compiled core of [`frc`](https://github.com/tiptenbrink/frc), a Python library for Fourier Ring Correlation (FRC) and resolution determination in microscopy. It was first developed for a Bachelor end project at TU Delft in 2021-2022. rustfrc has no dependencies: it takes any array that supports the buffer protocol, such as a NumPy array, an `array.array` or a `memoryview`, and returns a `memoryview`, which `numpy.asarray` wraps without copying it. For an API that takes and returns NumPy arrays, use `frc`.

rustfrc has wheels for Windows (x86_64), macOS (universal2, which runs on both Apple Silicon and Intel) and Linux (x86_64, glibc 2.28 and later). One wheel per platform covers CPython 3.11 and later, through the stable ABI, and one free-threaded CPython 3.15t and later, so a new Python version needs no new release. On other platforms, pip builds rustfrc from the sdist, which needs a Rust toolchain.

## Features

- `binom_split(a, seed=None)` replaces every count n of an array of non-negative integers with a sample k of binomial(n, 0.5), in an int32 array of the same shape. It counts the ones among n random bits, from 16 xoshiro256++ generators at once, with AVX-512 or AVX2 when the CPU has them. A Poisson-distributed count with mean λ becomes one with mean λ / 2, and so does n - k, independently of k.
- `one_frc(img, seed=None)` computes the FRC curve of a square image of counts from that one image: it splits every count with `binom_split`, Fourier transforms both halves, and correlates them over rings of equal distance from the origin. Ring r holds the frequencies f with r <= |f| < r + 1, as DIPlib's `RadialSum` bins them. The curve is a float64 array of length n // 2.
- `two_frc(img1, img2)` computes the FRC curve of two square images of one size.
- `ring_counts(n)` gives the number of frequencies in each ring of an n x n spectrum, as a uint64 array.
- `FrcPlan(n)` holds the FFT plans and the buffers for n x n images. `one_frc` and `two_frc` make one per call; for several images of one size, make one plan and use its `one_frc` and `two_frc` methods.

With a seed, the same input gives the same output on every CPU. Without one, every call draws a fresh seed from the operating system.

```python
import numpy as np
import rustfrc

img = np.random.default_rng(0).poisson(100, (1024, 1024))
curve = np.asarray(rustfrc.one_frc(img, seed=1))
```

## Performance

On 8 threads of an AMD EPYC 9V45 (Zen 5, AVX-512), with mean counts of 100 to 200:

- `binom_split`: its sampler takes ~1 ms on a 2000x2000 array, against ~49 ms for rustfrc 1.2.
- `FrcPlan.one_frc`: ~18 ms on a 4096x4096 image with a reused plan, ~30 ms with `one_frc`, which makes the plan, against ~1000 ms for `frc.one_frc` 0.1 (rustfrc 1.2's split with DIPlib's FFT and `RadialSum`).

## Installation

You can most easily install [rustfrc](https://pypi.org/project/rustfrc/) as follows:

```shell
pip install rustfrc
```

In a project managed by [uv](https://github.com/astral-sh/uv), add it with `uv add rustfrc`.

On a platform without a wheel, pip builds rustfrc from the sdist, which needs a Rust toolchain.

### From source (using maturin)

rustfrc uses [uv](https://github.com/astral-sh/uv) as its Python project manager and [maturin](https://pypi.org/project/maturin/) as its build backend. Build a wheel file from the project directory like this:

```shell
uvx maturin build --release
```

This builds one wheel for CPython 3.11 and later. To build the wheel for free-threaded Python, build with a free-threaded interpreter, e.g. `uvx maturin build --release -i python3.15t`.

This generates `.whl` files in `target/wheels`. Install one into a Python 3.11 or later environment with `pip install <.whl filename>`, for example `pip install rustfrc-2.0.0-cp311-abi3-win_amd64.whl`, and use `import rustfrc`. The only requirements are a Rust toolchain and Python for your platform.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions and can be uploaded to PyPI, take a look at `.github/workflows/wheels.yml`, which builds them with [pyo3/maturin-action](https://github.com/PyO3/maturin-action).

## Development

### uv

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Rust toolchain. Then run `uv sync` in the project directory. This creates a virtual environment in `.venv`, installs the (development) dependencies and builds the Rust extension with maturin. To develop against the oldest supported Python, pass it explicitly, e.g. `uv sync -p 3.11`.

After changing the Rust code, run `uv sync` again to rebuild the extension.

Run the Rust tests using `cargo test` and the Python tests using `uv run pytest`.

The Rust code is in `src`: `binomial.rs` holds the split, `frc.rs` the FRC, `buffers.rs` reads arrays through the buffer protocol and makes the `memoryview`s, and `lib.rs` holds the Python functions. The compiled module is `rustfrc._internal`, and `python/rustfrc/_internal.pyi` holds its types. `cargo bench` times the split and the FRC at several sizes.

### Releasing

Releases go through [touchgate](https://github.com/borink-org/touchgate):

1. Start the `touchgate` workflow with the version, and merge the pull request that it opens.
2. On the same day or the next, run `touchgate approve` in a checkout of rustfrc. It signs an approval of the tip of `main` with the security key, and starts `release.yml` with it.

If a release fails, fix the cause on `main` and run both steps again with the same version.

Every pull request adds a line under `## Unreleased` in `CHANGELOG.md`. `release.yml` publishes only from the `publish` branch, with the wheels that `wheels.yml` builds. A change to either file goes on `main`, and the repository owner copies it to `publish`.
