
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Hierarchical CNNs for Zooplankton Image Classification

<!-- badges: start -->

<!-- badges: end -->

This repository (will) provide several hierarchical CNNs for classifying
images of Zooplankton.

## Project Setup

This repo is structured as an R package at the root with a Python
subproject under `python/`. An R package (instead of a project) is used
intentionally to simplify the use of {testthat}, {devtools}, and related
development tools. But, you can think of this as just a project.

### R

Dependencies are managed with `renv`. Run `renv::restore()` from the R
console in the repo root directory to install the required dependencies.

Run R scripts from the console using `source()`,
e.g. `source("R/-hello/hello.R")`.

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

To confirm everything is installed correctly, run the following tests:

**R** (from the R console):

``` r
source("R/-hello/hello.R")
```

**Python** (from the terminal):

``` bash
uv run python/hello.py
```

Both should print the root directory of this project without errors or
warnings.

### Code Formatting

Formatting is enforced on commit via
[pre-commit](https://pre-commit.com/). R files and `.qmd` files are
formatted with [Air](https://posit-dev.github.io/air/), Python files
with [Ruff](https://docs.astral.sh/ruff/). Install `pre-commit` and set
up the hooks once:

``` bash
uv tool install pre-commit
pre-commit install
```

After that, formatting runs automatically on every commit. If files are
modified by the formatter, the commit will fail, after which you can
stage the modified files and commit again.
