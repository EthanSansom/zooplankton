import copy
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..utils import set_seed, EarlyStopper


class FlatModel(nn.Module):
    """
    Flat image classifier using a timm backbone.

    The backbone and classification head are split explicitly (rather than
    using timm's built-in num_classes) so that a trained FlatModel can be
    used as the backbone of an LCPNModel.

    Args:
        name:      Model name, used for saving and loading.
        directory: Root directory for saving model results, configuration, and metadata.
        n_classes: Number of output classes.
        config:    Config object with model, train, optimizer, scheduler sections.
    """

    def __init__(
        self,
        name: str,
        directory: Path,
        n_classes: int,
        config: Config,
    ):
        """
        Initialise backbone, head, history, and model metadata.

        The history dict is initialised with empty values here and reset
        at the start of fit().

        Args:
            name:      Model name, used for saving and loading.
            directory: Root directory for saving model artefacts.
            n_classes: Number of output classes.
            config:    Config object with model, train, optimizer, scheduler sections.
        """
        super().__init__()

        self.name = name
        self.directory = Path(directory)
        self.config = config
        self.backbone_name = config.model.backbone

        # The `backbone` and classifier `head` are manually split (rather than
        # specifying the head using `num_classes = n_classes`) to match the
        # implementation of `LCPNModel` and, in particular, so that a trained
        # `FlatModel` can easily be used as the backbone of a `LCPNModel`.
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=config.model.pretrained,
            num_classes=0,  # Remove classification head
            in_chans=config.model.in_chans,
        )
        self.feature_dim = self.backbone.num_features
        self.head = nn.Linear(self.feature_dim, n_classes)

        self.history = {
            "train": [],
            "valid": [],
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "epochs_completed": None,
        }
        self.model_metadata = {
            "name": self.name,
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "n_classes": n_classes,
            "n_parameters": self.get_num_parameters(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through backbone and classification head."""
        return self.head(self.backbone(x))

    # inference ----------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return predicted class indices for a batch of images.

        Args:
            x:       Input image tensor.
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped.

        Returns:
            Integer class index tensor of shape (batch_size,).
        """
        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)

    def predict_probabilities(
        self,
        x: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return softmax class probabilities for a batch of images.

        Args:
            x:       Input image tensor.
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped.

        Returns:
            Probability tensor of shape (batch_size, n_classes).
        """
        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)
            return torch.softmax(outputs, dim=1)

    # training -----------------------------------------------------------------

    def train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
    ) -> float:
        """
        Run one training epoch.

        Sets the model to train mode. Uses CrossEntropyLoss if no
        criterion is provided.

        Args:
            loader:    Training data loader.
            optimizer: Optimizer for parameter updates.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Mean loss per batch over the epoch.
        """
        self.train()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        device = self.config.metadata.device
        total_loss = 0.0

        for inputs, labels in tqdm(loader, desc="  Train", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(self.forward(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(
        self,
        loader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evaluate for one epoch, returning metrics only. Used during fit().
        This is a thin wrapper around evaluate().

        Args:
            loader:    Validation data loader.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Dict with keys "loss" and "accuracy".
        """
        metrics, _, _ = self.evaluate(loader, criterion, desc="  Valid")
        return metrics

    def test(
        self,
        loader,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Evaluate on a test set, returning metrics and collected predictions.
        This is a thin wrapper around evaluate().

        Args:
            loader:    Test data loader.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            metrics: Dict with keys "loss" and "accuracy".
            preds:   Predicted class indices, one per sample.
            true:    True class indices, one per sample.
        """
        metrics, preds, true = self.evaluate(
            loader, criterion, desc="  Test", collect_predictions=True
        )
        return metrics, preds, true

    def evaluate(
        self,
        loader,
        criterion: Optional[nn.Module] = None,
        desc: str = "  Eval",
        collect_predictions: bool = False,
    ) -> Tuple[Dict[str, float], Optional[List], Optional[List]]:
        """
        Evaluate the model over a data loader.

        Sets the model to eval mode. Used internally by validate() and test().

        Args:
            loader:              Data loader to evaluate on.
            criterion:           Optional loss function. Defaults to CrossEntropyLoss.
            desc:                tqdm progress bar label.
            collect_predictions: If True, return per-sample predictions and
                                 true labels. If False, these are returned as None.

        Returns:
            metrics: Dict with keys "loss" and "accuracy".
            preds:   Predicted class indices if collect_predictions, else None.
            true:    True class indices if collect_predictions, else None.
        """
        self.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        device = self.config.metadata.device
        total_loss = 0.0
        all_preds, all_true = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.forward(inputs)
                total_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(labels.cpu().tolist())

        n = len(all_preds)
        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": sum(p == t for p, t in zip(all_preds, all_true)) / n,
        }
        return (
            metrics,
            all_preds if collect_predictions else None,
            all_true if collect_predictions else None,
        )

    def fit(
        self,
        train_loader,
        valid_loader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict:
        """
        Train the model for the number of epochs specified in config.

        Resets history at the start of each call. The seed is incremented
        each epoch (cfg.train.seed + epoch) for reproducible but varied
        shuffling. Defaults to Adam and CosineAnnealingLR if no optimizer
        or scheduler are provided respectively.

        Args:
            train_loader: Training data loader.
            valid_loader: Validation data loader.
            criterion:    Optional loss function. Defaults to CrossEntropyLoss.
            optimizer:    Optional optimizer. Defaults to Adam with
                          cfg.optimizer.learning_rate.
            scheduler:    Optional LR scheduler. Defaults to CosineAnnealingLR
                          with cfg.scheduler.learning_rate_min.

        Returns:
            History dict with the following structure:
            {
                "train":            list of {"loss": float} per epoch,
                "valid":            list of {"loss": float, "accuracy": float} per epoch,
                "start_time":       str  # ISO 8601 datetime,
                "end_time":         str  # ISO 8601 datetime,
                "duration_seconds": float,
                "epochs_completed": int,
            }
        """
        cfg = self.config

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.optimizer.learning_rate,
                weight_decay=0,
            )
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.epochs,
                eta_min=cfg.scheduler.learning_rate_min,
            )

        self.history = {
            "train": [],
            "valid": [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "epochs_completed": None,
        }
        n_epochs = cfg.train.epochs
        start = time.time()

        best_state = None
        early_stopper = EarlyStopper(cfg.early_stop.patience, cfg.early_stop.min_delta)
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            set_seed(cfg.train.seed + epoch)

            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            self.history["train"].append({"loss": train_loss})
            print(f"  Train: loss={train_loss:.4f}")

            valid_metrics = self.validate(valid_loader, criterion)
            self.history["valid"].append(valid_metrics)
            print(
                f"  Valid: loss={valid_metrics['loss']:.4f}, accuracy={valid_metrics['accuracy']:.4f}"
            )

            early_stopper.step(valid_metrics["loss"])
            if epoch > cfg.early_stop.min_epochs and early_stopper.should_stop():
                print(
                    f"  Early stopping patience ({early_stopper.patience}) exceeded.\n"
                    f"  Stopped training early at epoch {epoch + 1}."
                )
                break
            elif early_stopper.is_best_epoch():
                best_state = copy.deepcopy(self.state_dict())

        self.history["duration_seconds"] = time.time() - start
        self.history["end_time"] = datetime.now().isoformat()
        self.history["stopped_early"] = early_stopper.stopped_early()
        self.history["epochs_completed"] = len(self.history["train"])
        self.history["best_epoch"] = early_stopper.best_epoch()

        # Restore model weights to the best state achieved during training
        if best_state is not None:
            self.load_state_dict(best_state)

        print(f"\nDone. ({self.history['duration_seconds']:.1f}s)")
        return self.history

    # read / write -------------------------------------------------------------

    def save(self, timestamp: bool = True, overwrite: bool = False) -> Path:
        """
        Save model weights, config, history, and metadata to disk.

        Creates a subdirectory under self.directory named after self.name,
        optionally suffixed with a timestamp. Saves:
        - weights.pth: model state dict
        - config.toml: training config
        - history.json: per-epoch training history, e.g. that returned by fit()
        - model_metadata.json: architecture metadata

        Args:
            timestamp: If True, appends a datetime stamp to the save directory
                       name to avoid overwriting prior runs.
            overwrite: If True, allows saving into an existing directory.

        Returns:
            Path to the save directory.
        """
        if timestamp:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.directory / f"{self.name}_{stamp}"
        else:
            save_dir = self.directory / self.name

        save_dir.mkdir(parents=True, exist_ok=overwrite)

        torch.save(self.state_dict(), save_dir / "weights.pth")
        self.config.save(save_dir / "config.toml")

        with open(save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        with open(save_dir / "model_metadata.json", "w") as f:
            json.dump(self.model_metadata, f, indent=2)

        print(f"Saved to {save_dir}")
        return save_dir

    @classmethod
    def load(cls, directory: Path) -> "FlatModel":
        """
        Load a FlatModel from a saved directory.

        Reconstructs the model from config.toml and model_metadata.json,
        loads weights from weights.pth, and restores history from history.json.

        Args:
            directory: Path to a directory created by save().

        Returns:
            Loaded FlatModel instance.
        """
        load_dir = Path(directory)
        config = Config(load_dir / "config.toml")

        with open(load_dir / "model_metadata.json") as f:
            model_metadata = json.load(f)

        model = cls(
            name=model_metadata["name"],
            directory=load_dir.parent,
            n_classes=model_metadata["n_classes"],
            config=config,
        )

        state_dict = torch.load(
            load_dir / "weights.pth",
            map_location=config.metadata.device,
        )
        model.load_state_dict(state_dict)

        with open(load_dir / "history.json") as f:
            model.history = json.load(f)

        return model

    # helpers ------------------------------------------------------------------

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Return parameter counts for the backbone, head, and total.

        Returns:
            Dictionary with keys "backbone", "heads", and "total".
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params,
        }

    def __repr__(self) -> str:
        """Return a string summary of the model architecture and parameter count."""
        param_counts = self.get_num_parameters()
        return (
            f"FlatModel(\n"
            f"  backbone={self.backbone_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  parameters={param_counts['total']:,}\n"
            f")"
        )
