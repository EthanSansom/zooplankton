# library(dplyr)
library(here)
library(reticulate)

# Ensure we're using the uv virtual environment
reticulate::use_virtualenv(here::here("python/.venv"), required = TRUE)

# Load a sample 100 images from each class of EMNIST (e.g. 0:9, A:Z, a:z)
py_run_string(
  "
import torch
import torchvision
import numpy as np
from pathlib import Path

np.random.seed(123)
torch.manual_seed(123)

data = torchvision.datasets.EMNIST(
    root=Path('python/00_raw_data'),
    split='byclass',
    train=True,
    download=False
)

n_per_class = 100
n_classes = len(data.classes)
images_list = []
labels_list = []

for class_idx in range(n_classes):
    class_indices = (data.targets.clone().detach() == class_idx).nonzero(as_tuple=True)[0]
    sampled = class_indices[torch.randperm(len(class_indices))[:n_per_class]]
    for idx in sampled:
        image, label = data[idx.item()]
        images_list.append(np.array(image).squeeze().T)
        labels_list.append(label)

images = np.stack(images_list)  # shape: (6200, 28, 28)
labels = np.array([data.classes[i] for i in labels_list])
"
)

# Python variables are stored in `reticulate::py`
images <- py$images
labels <- as.vector(py$labels) # An <array> by default

# Plot the first image
graphics::image(images[1, , ])
