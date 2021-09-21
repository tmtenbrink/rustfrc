# rustfrc

rustfrc is a Python package with some fast Rust functions for use with FRC resolution determination for microscopy. It is in development for use during a Bachelor end project for the TU Delft in 2021-2022.

Since rustfrc contains compiled extensions and is not pure Python, it is not available for all platforms, but only for those with available compiled wheels. As of version 0.1.2, they are available for Windows (x86_64), macOS (x86_64 and universal2) and Linux (x86_64). However, since Rust and Python are supported on many platforms, it is not difficult to compile for other platforms (see below).

## Installation

You can install [rustfrc](https://pypi.org/project/rustfrc/) as follows:

```shell
pip install rustfrc
```

### From source (using maturin)

rustfrc uses [poetry](https://github.com/python-poetry/poetry) as its Python dependency manager. For best results, create a `poetry` virtualenv (be sure to install virtualenv as a systems package) with the `pyproject.toml` and run `poetry update` to install the required packages. 
 Installing [maturin](https://pypi.org/project/maturin/) manually should also work. It is also necessary to have a Rust toolchain installed on your computer. Rust can be easily installed using [rustup](https://rustup.rs/).

Build a wheel file like this (if using poetry, append `poetry run` before the command) from the project directory:

```shell
maturin build --release
```

If you want to choose which versions of Python to build for, you can append e.g. `maturin build --release -i python3.9 python3.8 python3.7`.

This generates `.whl` files in `\target\wheels`. Then, create a Python environment of your choosing (with `numpy ^1.18` and `python ^3.7`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. This should be generally valid for all platforms. 

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions and can be uploaded to PyPI, using a [manylinux](https://github.com/pypa/manylinux) container is necessary. 

This example assumes a manylinux2014 (x86_64) target, using other docker base images should work to compile for other targets. Go into the `rustfrc/docker` directory and run:
```shell
docker pull quay.io/pypa/manylinux2014_x86_64
docker build -t tmtenbrink/manylinux2014-rustfrc --build-arg PY_ABI=cp39-cp39 .
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

#### macOS

Using the magic of GitHub actions (see the repository), it is possible to compile wheel files for macOS. Running the CI build workflow will create an artifact containing both universal2 (for modern macOS systems, including Apple Silicon Macs) and x86_64 (for older macOS versions) wheel files.