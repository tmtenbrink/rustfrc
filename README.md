# rustfrc

## Installation

### Wheel (Windows)

There is a Windows-compatible .whl file available in releases. Use `pip install <.whl filename>` in your Python environment, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`, to install it. Then, use `import rustfrc` in your Python script to be able to use the Rust functions.

### From source (using maturin)

`rustfrc` uses [`poetry`](https://github.com/python-poetry/poetry) as its Python dependency manager. For best results, create a `poetry` virtualenv with the `pyproject.toml` and run `poetry update` to install the required packages. 
Otherwise, installing [`maturin`](https://pypi.org/project/maturin/) manually should also work.

Build a wheel file like this:

```
maturin build --release
```

This generates a `.whl` file in `\target\wheels`. Then, create a Python environment of your choosing (with `numpy ^1.18` and `python ^3.7`), drop the `.whl` file in it and run `pip install <.whl filename>`, for example: `pip install rustfrc-0.1.0-cp39-none-win_amd64.whl`. Then, use `import rustfrc` in your Python script to be able to use the Rust functions. Building for Linux is a bit more challenging, take a look at the maturin page.

Take a look at [PyO3](https://github.com/PyO3/pyo3) for other installation options as the only true requirement for building is using a tool that understands PyO3 bindings, as those are used in the Rust code.
