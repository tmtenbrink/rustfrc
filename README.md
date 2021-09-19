# rustfrc

## Installation

### Wheel (Windows)

There is a Windows-compatible .whl file available in releases. Use `pip install <.whl filename>` in your Python environment, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`, to install it. Then, use `import rustfrc` in your Python script to be able to use the Rust functions.

### From source (using maturin)

`rustfrc` uses [`poetry`](https://github.com/python-poetry/poetry) as its Python dependency manager. For best results, create a `poetry` virtualenv with the `pyproject.toml` and run `poetry update` to install the required packages. 
Otherwise, installing [`maturin`](https://pypi.org/project/maturin/) manually should also work.

Build a wheel file like this (if using poetry, append `poetry run` before the command) from the project directory:

```shell
maturin build --release
```

If you want to choose which versions of Python to build for, you can append e.g. `-i python3.9 python3.8 python3.7`

This generates `.whl` files in `\target\wheels`. Then, create a Python environment of your choosing (with `numpy ^1.18` and `python ^3.7`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. Building for Linux is a bit more challenging, take a look at the maturin page.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.

#### Manylinux

If you want to build .whl files that are compatible with a wide range of Linux distributions, using a [manylinux](https://github.com/pypa/manylinux) container is necessary. 

Go into the `rustfrc/docker` directory and run:
```shell
docker pull quay.io/pypa/manylinux2014_x86_64
docker build -t tmtenbrink/manylinux2014-rustfrc --build-arg PY_ABI=cp39-cp39 .
```

After the image is built, run it as a container:
```shell
docker run -it tmtenbrink/manylinux2014-rustfrc
```

Then follow the same steps as above, but when the wheels are built do, which should resolve any issues: 
```shell
find . -name '*.whl' -exec auditwheel repair {} \;
```

#### Uploading to PyPI (test)

(Again, use `poetry run`)
```shell
twine upload --repository testpypi wheelhouse/*
```