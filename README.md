# rustfrc

![GitHub release](https://flat.badgen.net/github/release/tiptenbrink/rustfrc)
![License](https://flat.badgen.net/github/license/tiptenbrink/rustfrc)

rustfrc is a Python package with some fast Rust functions that are useful when performing Fourier Ring Correlation (FRC) computations for resolution determination in microscopy (specifically optical nanoscopy). It was originally developed for use in a Bachelor end project for the TU Delft in the period 2021-2022. See the Python package [`frc`](https://github.com/tiptenbrink/frc) for examples of its usage. The `test_split.py` file in the `tests` directory also holds some examples.

Since rustfrc contains compiled (Rust) extensions and is not pure Python, it is not available for all platforms, but only for those with available compiled wheels or a Rust toolchain and `maturin` support (see below). They are available for Windows (x86_64), macOS (universal2, which runs on both Apple Silicon and Intel) and Linux (x86_64, glibc 2.28 and later), for CPython 3.9 and later and for free-threaded CPython 3.15t and later. However, since Rust and Python are supported on many platforms, it is not difficult to compile for other platforms (see below).

## Features

rustfrc has only a few features. The primary one is `binom_split(x: ndarray) -> ndarray` which samples binomial _(n, 0.5)_ with n as the array element value. The operation is fully parallelized and somewhere between 3-10x faster than sampling using NumPy.

Furthermore, there are also (since version 1.1) `sqr_abs(a: ndarray) -> ndarray` and `pois_gen(lam: float, shape: tuple[int, ...]) -> ndarray`.

`sqr_abs` computes the element-wise norm and square of a complex array, while `pois_gen` generates an array of the specified size using the Poisson distribution and a single parameter λ.

## Requirements

* Python 3.9 or later, or free-threaded Python 3.15t or later
* NumPy 1.19.3+ (newer Python versions need newer NumPy, e.g. Python 3.13 requires NumPy 2.1+)

## Performance

On an i7-8750H, a decently high performance 6-core chip from 2017, I measured the following speeds:

- `binom_split`: ~210 ms on a 4000x4000 array, with each element Poisson-generated with a mean of 200
- `pois_gen`: ~420 ms to generate a 4000x4000 array with mean 200
- `sqr_abs`: ~40 ms on a 4000x4000 array, where each element is a complex number with both the real and imaginary parts having a mean of 200

Take this with a grain of salt, but it should provide a decent order of magnitude for larger images. 

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

This builds one wheel for CPython 3.9 and later. To build the wheel for free-threaded Python, build with a free-threaded interpreter, e.g. `uvx maturin build --release -i python3.15t`.

This generates `.whl` files in `/target/wheels`. Then, create a Python environment of your choosing (with `numpy>=1.19.3` and `python>=3.9`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-1.2.0-cp39-abi3-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. This should be generally valid for all platforms. The only real requirement is the availability of a Rust toolchain and Python for your platform.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions and can be uploaded to PyPI, take a look at `.github/workflows/wheels.yml`, which builds them with [pyo3/maturin-action](https://github.com/PyO3/maturin-action).

## Development

### uv

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Rust toolchain. Then run `uv sync` in the project directory. This creates a virtual environment in `.venv`, installs the (development) dependencies and builds the Rust extension with maturin. To develop against the oldest supported Python, pass it explicitly, e.g. `uv sync -p 3.9`.

After changing the Rust code, run `uv sync` again to rebuild the extension.

Run the Rust tests using `cargo test` and the Python tests using `uv run pytest`.

Note that type information is not available for the Rust functions, you will have to look at the Rust source code. Maturin builds a package structures as follows:
- root `rustfrc` package
    - `_internal`: this includes `binom_split_py`, etc.
    - `split`: this is the Python source code in `python/rustfrc/split.py`

### Releasing

Releases go through [touchgate](https://github.com/borink-org/touchgate):

1. Start the `touchgate` workflow with the version, and merge the pull request that it opens.
2. On the same day or the next, run `touchgate approve` in a checkout of rustfrc. It signs an approval of the tip of `main` with the security key, and starts `release.yml` with it.

If a release fails, fix the cause on `main` and run both steps again with the same version.

Every pull request adds a line under `## Unreleased` in `CHANGELOG.md`. `release.yml` publishes only from the `publish` branch, with the wheels that `wheels.yml` builds. A change to either file goes on `main`, and the repository owner copies it to `publish`.
