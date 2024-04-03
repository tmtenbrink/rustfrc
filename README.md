# rustfrc

![GitHub release](https://flat.badgen.net/github/release/tmtenbrink/rustfrc)
![License](https://flat.badgen.net/github/license/tmtenbrink/rustfrc)

rustfrc is a Python package with some fast Rust functions that are useful when performing Fourier Ring Correlation (FRC) computations for resolution determination in microscopy (specifically optical nanoscopy). It was originally developed for use in a Bachelor end project for the TU Delft in the period 2021-2022. See the Python package [`frc`](https://github.com/tmtenbrink/frc) for examples of its usage. The `test_split.py` file in the `tests` directory also holds some examples.

Since rustfrc contains compiled (Rust) extensions and is not pure Python, it is not available for all platforms, but only for those with available compiled wheels or a Rust toolchain and `maturin` support (see below). They are available for Windows (x86_64), macOS (x86_64 and universal2, which includes Apple Silicon) and Linux (x86_64). However, since Rust and Python are supported on many platforms, it is not difficult to compile for other platforms (see below).

## Features

rustfrc has only a few features. The primary one is `binom_split(x: ndarray) -> ndarray` which samples binomial _(n, 0.5)_ with n as the array element value. The operation is fully parallelized and somewhere between 3-10x faster than sampling using NumPy.

Furthermore, there are also (since version 1.1) `sqr_abs(a: ndarray) -> ndarray` and `pois_gen(lam: float, shape: tuple[int, ...]) -> ndarray`.

`sqr_abs` computes the element-wise norm and square of a complex array, while `pois_gen` generates an array of the specified size using the Poisson distribution and a single parameter λ.

## Requirements

* Python 3.8-3.12
* NumPy 1.18+ (exact version might depend on Python version, e.g. Python 3.12 requires NumPy 1.26)

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

However, for an optimal Python experience, use [poetry](https://github.com/python-poetry/poetry) and install it using `poetry add rustfrc`.

### From source (using maturin)

rustfrc uses [poetry](https://github.com/python-poetry/poetry) as its Python dependency manager. For best results, create a `poetry` virtualenv (be sure to install virtualenv as a systems package) with the `pyproject.toml` and run `poetry update` to install the required packages. I recommend installing [maturin](https://pypi.org/project/maturin/) as a global tool (e.g. with `cargo binstall`).

Build a wheel file like this (if using poetry, append `poetry run` before the command) from the project directory:

```shell
maturin build --release
```

If you want to choose which versions of Python to build for, you can write e.g. `maturin build --release -i python3.9 python3.8 python3.7`. Here, for example '`python3.7`' should be an available Python command installed on your computer.

This generates `.whl` files in `/target/wheels`. Then, create a Python environment of your choosing (with `numpy ^1.18` and `python ^3.7`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. This should be generally valid for all platforms. The only real requirement is the availability of a Rust toolchain and Python for your platform.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions and can be uploaded to PyPI, take a look at the GitHub Actions files and [pyo3/maturin-action](https://github.com/PyO3/maturin-action).
