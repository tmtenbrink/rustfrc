Time to beat:

110 ms for generating 4000x4000 (i.e. 16M) binomials (parallelized)
- i.e. 160M binomials per second (~6 ns per binomial), where each has a different input parameter

Currently we can:
- generate 40M random u64's per second (16.8M in 415 ms)
- However, nearly 120 ms is device setup, 120 ms is setting up buffers and 100 ms is casting it to the right type
- So only 50 ms spent in the GPU doing compute

vs. cpu thread rng: 200M u64's per second (non-parallelized)

Take a look at AES hardware acceleration? https://docs.rs/aes/latest/aes/

Take a look at cupy https://cupy.dev/

https://dimitri.xyz/random-ints-from-random-bits/

# rustfrc

![GitHub release](https://flat.badgen.net/github/release/tmtenbrink/rustfrc)
![License](https://flat.badgen.net/github/license/tmtenbrink/rustfrc)

rustfrc is a Python package with some fast Rust functions that are using when performing Fourier Ring Correlation (FRC) computations for resolution determination in microscopy (specifically optical nanoscopy). It is in development for use in a Bachelor end project for the TU Delft in the period 2021-2022.

Since rustfrc contains compiled (Rust) extensions and is not pure Python, it is not available for all platforms, but only for those with available compiled wheels or a Rust toolchain and `maturin` support (see below). They are available for Windows (x86_64), macOS (x86_64 and universal2, which includes Apple Silicon) and Linux (x86_64). However, since Rust and Python are supported on many platforms, it is not difficult to compile for other platforms (see below).

## Features

Currently, rustfrc does not have many features. The primary one is `binom_split(x: ndarray) -> ndarray` which samples binomial _(n, 0.5)_ with n as the array element value. The operation is fully parallelized and somewhere between 3-10x faster than sampling using NumPy.

Furthermore, there are also (since version 1.1) `sqr_abs(a: ndarray) -> ndarray` and `pois_gen(lam: float, shape: tuple[int, ...]) -> ndarray`.

`sqr_abs` computes the element-wise norm and square of a complex array, while `pois_gen` generates an array of the specified size using the Poisson distribution and a single parameter λ.

## Requirements

* Python 3.7 or greater
* NumPy 1.18 or greater

## Installation

You can most easily install [rustfrc](https://pypi.org/project/rustfrc/) as follows:

```shell
pip install rustfrc
```

However, for an optimal Python experience, use [poetry](https://github.com/python-poetry/poetry) and install it using `poetry add rustfrc`.

### From source (using maturin)

rustfrc uses [poetry](https://github.com/python-poetry/poetry) as its Python dependency manager. For best results, create a `poetry` virtualenv (be sure to install virtualenv as a systems package) with the `pyproject.toml` and run `poetry update` to install the required packages. 
 Installing [maturin](https://pypi.org/project/maturin/) manually should also work. It is also necessary to have a Rust toolchain installed on your computer. Rust can be easily installed using [rustup](https://rustup.rs/).

Build a wheel file like this (if using poetry, append `poetry run` before the command) from the project directory:

```shell
maturin build --release
```

If you want to choose which versions of Python to build for, you can write e.g. `maturin build --release -i python3.9 python3.8 python3.7`. Here, for example '`python3.7`' should be an available Python command installed on your computer.

This generates `.whl` files in `/target/wheels`. Then, create a Python environment of your choosing (with `numpy ^1.18` and `python ^3.7`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. This should be generally valid for all platforms. The only real requirement is the availability of a Rust toolchain and Python for your platform.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions and can be uploaded to PyPI, using a [manylinux](https://github.com/pypa/manylinux) container is necessary. 

This example assumes a manylinux2014 (x86_64) target, using other docker base images should work to compile for other targets. Go into the `rustfrc/docker` directory and run:
```shell
docker pull quay.io/pypa/manylinux2014_x86_64
docker build --no-cache -t tmtenbrink/manylinux2014-rustfrc --build-arg PY_ABI=cp39-cp39 .
```

After the image is built, run it as a container:
```shell
docker run -it tmtenbrink/manylinux2014-rustfrc
```

The repository should now be installed at `/opt/rustfrc`, with its dependencies and `maturin` installed. Then follow the same steps as above, but when the wheels are built run the following, which should resolve any issues: 
```shell
find . -name '*.whl' -exec auditwheel repair {} \;
```

The complete wheel files can now be found in a `/wheelhouse` directory.

Take a look at the GitHub Actions for this project for an idea of how to automate this.