import torch
from torch.utils.data import random_split
import numpy as np

import os
import random


def set_seed(seed=123):
    # Seed python, numpy, and torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Metal Performance Shaders (MPS) on Mac
    # See: https://developer.apple.com/metal/pytorch/
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # CUDA-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split(data, lengths, cfg):
    return random_split(
        data,
        lengths,
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
