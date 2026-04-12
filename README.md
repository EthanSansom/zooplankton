
# Hierarchical CNNs for Zooplankton Image Classification

This repository provides a Python library for training and evaluating hierarchical 
CNNs for image classification, applied to a non-publicly-available dataset of
freshwater Zooplankton images and to the EMNIST[^1] (Extended-MNIST) character 
dataset (62 classes A-Z, a-z, 0-9).

The EMNIST dataset is available for download via the `torchvision.datasets` module. 
The Zooplankton images were provided by the Ontario Ministry of Natural Resources 
(OMNR).

The repository includes implementations of flat CNN baselines and a Local Classifier 
Per Parent Node[^2] (LCPN) architecture using a morphological hierarchy (for EMNIST)
and a taxonomic hierarchy (for Zooplankton).

## Directory Structure

This repository bundles a shared library for training and evaluating flat and
LCPN CNN classifiers under `python/cnn`, alongside experiments/training-demos on 
the EMNIST dataset (`python/emnist`) and Zooplankton dataset (`python/zooplankton`).

```
python/
├── cnn/src/cnn/       # Shared Library
│   ├── models/
│   │   ├── flat.py
│   │   └── hierarchical.py
│   ├── config.py
│   ├── data.py
│   ├── hierarchy.py
│   ├── label_map.py
│   ├── metrics.py
│   └── utils.py
├── emnist/             # EMNIST demos
│   ├── 00_configs/
│   ├── 00_hierarchies/
│   ├── 00_raw_data/    # gitignored
│   ├── 01_results/     # gitignored
│   └── 99_demos/
└── zooplankton/        # Zooplankton demos and experiments
    ├── 00_configs/
    ├── 00_hierarchies/
    ├── 00_label_maps/
    ├── 00_raw_data/    # gitignored
    ├── 01_results/     # gitignored
    ├── 97_experiments/
    ├── 98_eda/
    └── 99_demos/
```

## Architecture (Zooplankton)

See `resources/98_slides/presentation_2026_04_08.pdf` or `resources/99_report/report.pdf` for a more detailed description of the model architectures and associated experiments.

### Hierarchy

Currently, Zooplankton are classified by the following taxonomic hierarchy.

![Zooplankton hierarchy](resources/00_images/zooplankton_hierarchy.png)

Shown below are examples of the 11 leaf classes:

![Zooplankton classes](resources/00_images/zooplankton_leaf_classes.png)

### Flat Classifier

The flat classifier consists of a ResNet18 feature extractor and a single 
classification head, which produces a probability distribution over all 11 
leaf classes.

![Flat classifier architecture](resources/00_images/flat_zooplankton_architecture.png)

### LCPN Classifier

The LCPN classifier consists of a ResNet18 feature extractor and five classification 
heads, one per parent node in the hierarchy. Each head produces a probability 
distribution over its immediate children.

![LCPN classifier architecture](resources/00_images/lcpn_zooplankton_architecture.png)

## Library: `cnn` package

The package is currently designed for internal use to support the experiments in 
this repository, but may be made more robust and well-documented in the future, 
for external use.

- `config.py`: `Config` class for loading and accessing TOML-based hyperparameter files.
- `data.py`: `ImageDataset` for flat models; `LCPNDataset` and `LCPNCollator` for hierarchical labelling and batching.
- `hierarchy.py`: `Hierarchy` class for loading, validating, and querying JSON hierarchy files.
- `label_map.py`: `LabelMap` class for loading, validating, and querying JSON directory to class label mapping files.
- `metrics.py`: Miscellaneous flat and hierarchical classifier metrics functions.
- `utils.py`: `set_seed` and `split` for reproducible train/validation/test partitioning.
- `models/flat.py`: `FlatModel`, a flat CNN classifier built on a pretrained [timm](https://pypi.org/project/timm/) backbone (ResNet18 default). 
- `models/hierarchical.py`: `LCPNModel`, a LCPN architecture with one classification head per parent node. Supports greedy and globally optimal prediction and loading backbone weights from a trained `FlatModel`.

## Running Experiments

### Demo Experiments

From the project directory:
```bash
# Zooplankton (data unavailable)
uv run python/zooplankton/97_experiments/flat.py
uv run python/zooplankton/97_experiments/lcpn.py
uv run python/zooplankton/97_experiments/lcpn_flat_backbone.py

# EMNIST demos (data available)
uv run python/emnist/99_demos/01_flat_model.py
uv run python/emnist/99_demos/01_lcpn_model.py
uv run python/emnist/99_demos/02_lcpn_flat_backbone_model.py
```

Each script reads its configuration from the corresponding TOML file in the sibling
directory `00_configs/` and writes model weights, configuration, and metrics to a directory 
in `01_results/`. Configuration files are used to determine the random seed for
an experiment and model hyperparameters such as the learning rate and early-stopping
criteria. See `python/zooplankton/00_configs/demo_lcpn.toml` for an example of the
required configuration parameters.

The `load()` method of the `FlatModel` and `LCPNModel` classes supports re-loading 
a trained model from its save directory in `01_results/`.

### LCPN Experiments

LCPN training scripts additionally require a JSON hierarchy file
under `00_hierarchies/` and JSON class-label map under `00_label_maps/`.

The `zooplankton/00_raw_data/` directory (or any raw data directory used for
training LCPN models) must be structured as follows:

```
00_raw_data/
├── class1/
│   ├── class11.tif
│   ├── ...
│   └── class1k.tif
├── ...
└── classN/
    ├── classN1.tif
    ├── ...
    └── classNj.tif
```
Where the raw class labels are encoded in the directory names and all 
images are stored as `.tif` files.

A label map file (e.g. `zooplankton/00_label_maps/demo_lcpn.json`) maps these `00_raw_data/`
subdirectories to the set of classes that you wish to classify:

```
{
  "class12": ["class1", "class2"], # Treat two directories as one class 
  "class4": ["class4"],            # Exclude a directory (e.g. skip "class3")
  ...,
  "classN": ["classN"]
}
```

Finally, a hierarchy file (e.g. `zooplankton/00_hierarchies/demo_taxonomic.json`)
maps these classes to nodes on a hierarchy:

```
{
    "root": ["lt_5", "gte_5"],
    "lt_5": ["class12", "class4"],
    "gte_5": ["even", "odd"],
    "even": ["class6", "class8", "classN"],
    "odd": ["class7", "class9"]
}
```

This hierarchy indicates that *class12* images have a hierarchical label of
\[*lt_5*, *class12*\] while *class9* images have a hierarchical label of 
\[*gte_5*, *odd*, *class9*\].

Scripts under `zooplankton/97_experiments/` and `zooplankton/99_demos/` are written
such that all modifications to the model training parameters and hierarchy
are made by specifying the appropriate configuration files under the `# User settings` 
section at the top of each script:

```python
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset, LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy
from cnn.label_map import LabelMap
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

CONFIG_FILE = "demo_lcpn.toml"          # Configuration (e.g. random seed)
HIERARCHY_FILE = "demo_taxonomic.json"  # Hierarchy
LABEL_MAP_FILE = "demo_lcpn.json"       # Directory to label map
MODEL_NAME = "demo_lcpn"                # Name used for the results directory

# Remainder of the script...
```

## Project Setup

### Python

This project requires Python 3.12. Dependencies are managed with `uv`. Run the 
following in the terminal at the repo root:

``` bash
uv sync --project python
```

This creates `python/.venv` and installs the required python dependencies.

### Check Your Installation

To confirm everything is installed correctly you may run the following demo
script:

``` bash
uv run python/emnist/99_demos/01_flat_model.py
```

If the installation is in working order, this script will download the EMNIST
image dataset to the `python/emnist/00_raw_data` directory, train a flat multi-class
CNN classifier on the 62 EMNIST classes, and save the results to a directory:
`python/emnist/01_results/demo_flat`.

To train a demo LCPN classifier instead, run: 

``` bash
uv run python/emnist/99_demos/01_lcpn_model.py
```

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

[^1]: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: Extending MNIST to handwritten letters. Proceedings of the International Joint Conference on Neural Networks (IJCNN). https://doi.org/10.1109/IJCNN.2017.7966217

[^2]: Silla, C.N., Freitas, A.A. A survey of hierarchical classification across different application domains. Data Min Knowl Disc 22, 31–72 (2011). https://doi.org/10.1007/s10618-010-0175-9
