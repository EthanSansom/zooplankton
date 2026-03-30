import os
import random

import numpy as np
import torch
from torch.utils.data import random_split

# helpers ----------------------------------------------------------------------


def set_seed(seed=123):
    """
    Seed Python, NumPy, and PyTorch (CPU, CUDA, and MPS) for reproducibility.
    """

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
    """
    Split a dataset into subsets using a seeded generator (cfg.train.seed).
    This is a thin wrapper around torch.utils.data.random_split().
    """

    return random_split(
        data,
        lengths,
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )


# classes ----------------------------------------------------------------------


class EarlyStopper:
    """
    Stops training early if a monitored metric does not improve for a given
    number of epochs.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """
        Args:
            patience:  Number of epochs without improvement before stopping.
            min_delta: Minimum change in the monitored metric to count as an improvement.
                       Negative for decreasing metrics (e.g. -0.01 for loss),
                       positive for increasing metrics (e.g. 0.01 for accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf") if min_delta <= 0 else float("-inf")
        self.epochs_without_improvement = 0
        self.epoch = 0

    def step(self, metric: float):
        """
        Update state per epoch.

        Args:
            metric: Value of the monitored metric for the current epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        self.epoch += 1
        improved = (
            metric < self.best_value + self.min_delta
            if self.min_delta <= 0
            else metric > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def is_best_epoch(self) -> bool:
        """
        True if the current step was the best yet.
        This should be called after step().
        """
        return self.epochs_without_improvement == 0

    def should_stop(self) -> bool:
        """
        Whether patience has been exceeded and training should stop.
        This should be called after step().
        """
        return self.epochs_without_improvement > self.patience

    def best_epoch(self) -> int:
        """The epoch at which the best metric value was recorded (1-indexed)."""
        return self.epoch - self.epochs_without_improvement

    def stopped_early(self) -> bool:
        """True if early stopping has been triggered."""
        return self.epochs_without_improvement >= self.patience
