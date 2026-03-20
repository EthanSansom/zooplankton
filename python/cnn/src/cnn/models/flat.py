import time
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path

import timm
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..utils import set_seed


class FlatModel(nn.Module):
    """Flat image classifier using a timm backbone."""

    def __init__(
        self,
        name: str,
        directory: Path,
        n_classes: int,
        config: Config,
    ):
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
        return self.head(self.backbone(x))

    # inference ----------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return predicted class indices for a batch of images."""
        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)

    def predict_probabilities(
        self,
        x: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return softmax class probabilities for a batch of images."""
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

    def evaluate(
        self,
        loader,
        criterion: Optional[nn.Module] = None,
        desc: str = "  Eval",
        collect_predictions: bool = False,
    ) -> Tuple[Dict[str, float], Optional[List], Optional[List]]:
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

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            set_seed(cfg.train.seed + epoch)

            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            self.history["train"].append({"loss": train_loss})
            print(f"  Train: loss={train_loss:.4f}")

            valid_metrics, _, _ = self.evaluate(valid_loader, criterion)
            self.history["valid"].append(valid_metrics)
            print(
                f"  Valid: loss={valid_metrics['loss']:.4f}, accuracy={valid_metrics['accuracy']:.4f}"
            )

        self.history["duration_seconds"] = time.time() - start
        self.history["end_time"] = datetime.now().isoformat()
        self.history["epochs_completed"] = len(self.history["train"])
        print(f"\nDone. ({self.history['duration_seconds']:.1f}s)")
        return self.history

    # read / write -------------------------------------------------------------

    def save(self, timestamp: bool = True, overwrite: bool = False) -> Path:
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
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params,
        }

    def __repr__(self) -> str:
        param_counts = self.get_num_parameters()
        return (
            f"FlatModel(\n"
            f"  backbone={self.backbone_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  parameters={param_counts['total']:,}\n"
            f")"
        )
