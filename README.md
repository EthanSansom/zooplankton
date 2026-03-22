
# Hierarchical CNNs for Zooplankton Image Classification

This repository (will) provide several hierarchical CNNs for classifying
images of Zooplankton.

## Project Setup

### Python

Dependencies are managed with `uv`. Run the following in the terminal at
the repo root:

``` bash
uv sync --project python
```

This creates `python/.venv` and installs the required python
dependencies.

Run Python scripts from the terminal using `uv`,
e.g. `uv run python/hello.py`.

### Check Your Installation

To confirm everything is installed correctly, run the following test:

``` bash
uv run python/hello.py
```

Both should print the root directory of this project without errors or
warnings.

### Code Formatting

Formatting is enforced on commit via
[pre-commit](https://pre-commit.com/). Python files are checked and re-formatted 
using [Ruff](https://docs.astral.sh/ruff/). Install `pre-commit` and set
up the hooks once:

``` bash
uv tool install pre-commit
pre-commit install
```

After that, formatting runs automatically on every commit. If files are
modified by the formatter, the commit will fail, after which you can
stage the modified files and commit again.
