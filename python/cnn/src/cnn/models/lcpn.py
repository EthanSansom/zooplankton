import copy
from datetime import datetime
import json
from pathlib import Path
import time
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..data import LCPNCollator
from ..hierarchy import Hierarchy
from ..utils import set_seed, EarlyStopper


class LCPNModel(nn.Module):
    """
    Local Classifier Per Parent Node (LCPN) hierarchical image classifier.

    Uses a shared backbone (feature extractor) with one linear classification
    head per parent node in the hierarchy. The backbone and heads are trained
    jointly unless the backbone is frozen.

    Args:
        name:      Model name, used for saving and loading.
        directory: Root directory for saving model artefacts.
        hierarchy: Hierarchy object defining the tree structure.
        config:    Config object with model, train, optimizer, scheduler sections.
    """

    # initialization -----------------------------------------------------------

    def __init__(
        self,
        name: str,
        directory: Path,
        hierarchy: Hierarchy,
        config: Config,
    ):
        """
        Initialise backbone, classification heads, history, and model metadata.

        If config.model.backbone_model is set, backbone weights are loaded from
        the specified FlatModel directory, overriding any timm pretrained weights.
        If config.model.freeze_backbone is True, backbone parameters are frozen
        and only the heads are trained. A warning is raised if both
        backbone_model and pretrained=True are set simultaneously.

        The history dict is initialised with empty values here and reset
        at the start of fit().

        Args:
            name:      Model name, used for saving and loading.
            directory: Root directory for saving model artefacts.
            hierarchy: Hierarchy object defining the tree structure.
            config:    Config object with model, train, optimizer, scheduler sections.
        """
        super().__init__()

        self.name = name
        self.directory = Path(directory)
        self.hierarchy = hierarchy
        self.config = config
        self.backbone_name = config.model.backbone

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=config.model.pretrained,
            num_classes=0,  # Remove classification head
            in_chans=config.model.in_chans,
        )
        self.feature_dim = self.backbone.num_features

        if config.model.backbone_model:
            self._load_backbone_weights(Path(config.model.backbone_model))

        if config.model.freeze_backbone:
            self.freeze_backbone()

        if config.model.backbone_model and config.model.pretrained:
            warnings.warn(
                "backbone_model is set, so pretrained=True has no effect, "
                "timm weights will be overwritten by the provided model weights.",
                UserWarning,
            )

        self.heads = nn.ModuleDict()
        for parent_node in hierarchy.get_parent_nodes():
            n_classes = hierarchy.num_children(parent_node)
            self.heads[parent_node] = nn.Linear(self.feature_dim, n_classes)

        # Empty training history, overwritten by `fit()`
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
            "n_heads": len(self.heads),
            "n_parameters": self.get_num_parameters(),
        }

    # forward ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            Dictionary mapping parent node names to logits, structured like:
            {
                'root': tensor of shape (batch_size, n_classes_root),
                'parent': tensor of shape (batch_size, n_classes_parent),
                'child_1': tensor of shape (batch_size, n_classes_child_1),
                'child_2': tensor of shape (batch_size, n_classes_child_2),
                ...
            }
        """
        features = self.backbone(x)
        return {node_name: head(features) for node_name, head in self.heads.items()}

    # backbone -----------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_is_frozen(self) -> bool:
        """Return True if backbone parameters are frozen."""
        return not next(self.backbone.parameters()).requires_grad

    # inference ----------------------------------------------------------------

    def predict_greedy(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Predict by greedily following the argmax at each parent node.

        At each level, the child with the highest logit is selected, and
        prediction continues until a leaf is reached. Provides the prediction
        for every parent node and the leaf node (e.g. ['root', 'curved', 'open_curve', 'C']).
        Use predict_greedy_fast() if paths are not needed (e.g. only class 'C'
        is required).

        Args:
            x:       Input image tensor of shape (batch_size, C, H, W).
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped. Useful for avoiding redundant
                     forward passes when predictions and probabilities are both needed.

        Returns:
            predictions: Predicted leaf node name per sample, e.g. ['C', 'O', 'A'].
            paths:       Root-to-leaf path per sample, e.g. [['root', 'curved', 'open_curve', 'C'], ...].
        """

        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)

            predictions, paths = [], []

            for i in range(x.shape[0]):
                current_node = self.hierarchy.root
                path = [current_node]

                # Iterate through each parent node on the hierarchy and use
                # the corresponding local classifier. If the `grandparent`
                # classifier emits logits = [1, 10, 2] then the `parent`
                # classifier at logits[1] == 10 is used as the next classifier.
                while current_node in self.hierarchy.parent_to_children:
                    logits = outputs[current_node][i]
                    pred_index = torch.argmax(logits).item()
                    children = self.hierarchy.parent_to_children[current_node]

                    if pred_index >= len(children):
                        raise ValueError(
                            f"Predicted index {pred_index} exceeds the number "
                            f"of child classes {len(children)}."
                        )

                    current_node = children[pred_index]
                    path.append(current_node)

                predictions.append(current_node)
                paths.append(path)

        return predictions, paths

    def predict_greedy_fast(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Predict by greedily following the argmax at each parent node.

        Faster variant of predict_greedy() that omits path tracking.
        Used internally during evaluate().

        Args:
            x:       Input image tensor of shape (batch_size, C, H, W).
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped.

        Returns:
            Predicted leaf node name per sample, e.g. ['C', 'O', 'A'].
        """
        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)

            predictions = []

            for i in range(x.shape[0]):
                current_node = self.hierarchy.root

                while current_node in self.hierarchy.parent_to_children:
                    logits = outputs[current_node][i]
                    pred_index = torch.argmax(logits).item()
                    children = self.hierarchy.parent_to_children[current_node]

                    if pred_index >= len(children):
                        raise ValueError(
                            f"Predicted index {pred_index} exceeds the number "
                            f"of child classes {len(children)}."
                        )

                    current_node = children[pred_index]

                predictions.append(current_node)

        return predictions

    def predict_global(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Predict by selecting the leaf with the highest joint probability
        across the full hierarchy.

        Unlike predict_greedy(), this considers all root-to-leaf paths and
        selects the globally optimal leaf rather than making locally optimal
        decisions at each node.

        Args:
            x:       Input image tensor of shape (batch_size, C, H, W).
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped.

        Returns:
            predictions: Predicted leaf node name per sample, e.g. ['C', 'O', 'A'].
                paths:   Root-to-leaf path per sample, e.g. [['root', 'curved', 'open_curve', 'C'], ...].
        """
        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)

            probs = self.prediction_probabilities(x, outputs=outputs)
            leaf_probs = probs["leaves"]  # {leaf_name: (batch_size,)}

            batch_size = x.shape[0]
            predictions, paths = [], []

            for i in range(batch_size):
                best_leaf = max(leaf_probs, key=lambda leaf: leaf_probs[leaf][i].item())
                predictions.append(best_leaf)
                paths.append(self.hierarchy.get_path_to_root(best_leaf))

        return predictions, paths

    def prediction_probabilities(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute softmax probabilities for all parent nodes and joint
        probabilities for all leaf nodes.

        Leaf probabilities are computed as the product of softmax probabilities
        along the root-to-leaf path.

        Args:
            x:       Input image tensor of shape (batch_size, C, H, W).
            outputs: Optional pre-computed forward pass output. If provided,
                     the forward pass is skipped.

        Returns:
            Returns a Dict with "root" and "leaves" keys, structured like:
            {
                "parents": {
                    "root":     tensor of shape (batch_size, n_children_root),
                    "curved":   tensor of shape (batch_size, n_children_curved),
                    ...
                },
                "leaves": {
                    "C": tensor of shape (batch_size,),
                    "O": tensor of shape (batch_size,),
                    ...
                },
            }
        """

        with torch.no_grad():
            if outputs is None:
                outputs = self.forward(x)

            batch_size = x.shape[0]

            parent_probs = {}
            for node_name, logits in outputs.items():
                parent_probs[node_name] = torch.softmax(logits, dim=1)

            leaf_probs = {}
            for leaf in self.hierarchy.get_leaf_nodes():
                path = self.hierarchy.get_path_to_root(leaf)

                leaf_prob = torch.ones(batch_size, device=x.device)

                for i in range(len(path) - 1):
                    parent = path[i]
                    child = path[i + 1]
                    child_index = self.hierarchy.get_child_index(parent, child)
                    leaf_prob *= parent_probs[parent][:, child_index]

                leaf_probs[leaf] = leaf_prob

        return {"parents": parent_probs, "leaves": leaf_probs}

    # training -----------------------------------------------------------------

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute total loss by summing per-parent-node losses over the batch.

        Labels use -1 as a sentinel for off-path samples (i.e. samples that
        do not pass through a given parent node). These are masked out before
        computing the loss for that classifier. Uses CrossEntropyLoss if no
        criterion is provided.

        Args:
            outputs:   Forward pass output from `forward()`, mapping parent node
                       names to logit tensors of shape (batch_size, n_children).
            labels:    Batched labels from `LCPNCollator`, mapping parent node
                       names to label tensors of shape (batch_size,), with -1
                       for off-path samples.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Scalar total loss tensor.
        """

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        total_loss = torch.tensor(0.0, device=self.config.metadata.device)
        for parent_classifier, child_logits in outputs.items():
            child_labels = labels[parent_classifier]
            mask = child_labels != -1

            if mask.sum() == 0:
                continue

            total_loss = total_loss + criterion(child_logits[mask], child_labels[mask])

        return total_loss

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
            loader:    Training data loader yielding (images, labels) batches,
                       where labels is a dict from LCPNCollator.
            optimizer: Optimizer for parameter updates.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Mean loss per batch over the epoch.
        """
        self.train()
        device = self.config.metadata.device
        total_loss = 0.0

        for inputs, labels in tqdm(loader, desc="  Train", leave=False):
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(
        self,
        loader,
        collator: LCPNCollator,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evaluate for one epoch, returning metrics only. Used during fit().
        This is a thin wrapper around evaluate().

        Args:
            loader:    Validation data loader.
            collator:  LCPNCollator used to decode true labels.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Dictionary with keys "loss" and "accuracy".
        """
        metrics, _, _ = self.evaluate(loader, collator, criterion, desc="  Valid")
        return metrics

    def test(
        self,
        loader,
        collator: LCPNCollator,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[
        Dict[str, float], List[str], List[str], List[List[str]], List[List[str]]
    ]:
        """
        Evaluate on a test set, returning metrics, predictions, and full paths.

        Unlike evaluate(), uses predict_greedy() to collect full root-to-leaf
        paths for all samples. Accuracy is computed over fully-labelled samples
        only. Paths are collected for all samples, including partially-labelled.

        Args:
            loader:    Test data loader.
            collator:  LCPNCollator used to decode true labels.
            criterion: Optional loss function. Defaults to CrossEntropyLoss.

        Returns:
            Tuple of:
                metrics:    Dict with keys "loss" and "accuracy".
                preds:      Predicted leaf node names, one per fully-labelled sample.
                true:       True leaf node names, one per fully-labelled sample.
                pred_paths: Full predicted root-to-leaf paths, one per sample.
                true_paths: Full true root-to-node paths, one per sample.
        """
        self.eval()
        device = self.config.metadata.device
        total_loss = 0.0
        all_preds, all_true = [], []
        all_pred_paths, all_true_paths = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="  Test", leave=False):
                inputs = inputs.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = self.forward(inputs)
                total_loss += self.compute_loss(outputs, labels, criterion).item()

                _, pred_paths = self.predict_greedy(inputs, outputs=outputs)
                true_leaves, is_leaf = collator.uncollate_label_leaves(labels)
                true_paths = collator.uncollate_label_paths(labels)

                for pred_path, true_leaf, true_path, leaf in zip(
                    pred_paths, true_leaves, true_paths, is_leaf
                ):
                    all_pred_paths.append(pred_path)
                    all_true_paths.append(true_path)
                    if leaf:
                        all_preds.append(pred_path[-1])
                        all_true.append(true_leaf)

        n = len(all_preds)
        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": sum(p == t for p, t in zip(all_preds, all_true)) / n,
        }
        return metrics, all_preds, all_true, all_pred_paths, all_true_paths

    def evaluate(
        self,
        loader,
        collator: LCPNCollator,
        criterion: Optional[nn.Module] = None,
        desc: str = "  Eval",
        collect_predictions: bool = False,
    ) -> Tuple[Dict[str, float], Optional[List[str]], Optional[List[str]]]:
        """
        Evaluate the model over a data loader.

        Sets the model to eval mode. Used internally by validate() and test().
        Partially-labelled samples (where the true label is not a leaf node)
        are excluded from accuracy computation.

        Args:
            loader:              Data loader yielding (images, labels) batches,
                                 where labels is a dict from LCPNCollator.
            collator:            LCPNCollator used to decode true labels.
            criterion:           Optional loss function. Defaults to CrossEntropyLoss.
            desc:                tqdm progress bar label.
            collect_predictions: If True, return per-sample predictions and
                                 true labels. If False, these are returned as None.

        Returns:
            metrics: Dict with keys "loss" and "accuracy".
            preds:   Predicted leaf node names if collect_predictions, else None.
            true:    True leaf node names if collect_predictions, else None.
        """
        self.eval()
        device = self.config.metadata.device
        total_loss = 0.0
        all_preds, all_true = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs = inputs.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = self.forward(inputs)
                total_loss += self.compute_loss(outputs, labels, criterion).item()

                preds = self.predict_greedy_fast(inputs, outputs=outputs)
                true, is_leaf = collator.uncollate_label_leaves(labels)

                # predict_greedy_fast() will always return a leaf node prediction,
                # however some input samples may be partially labeled (i.e. have
                # no "leaf" class). These partially labelled samples are excluded
                # from evaluation.
                all_preds.extend(pred for (pred, leaf) in zip(preds, is_leaf) if leaf)
                all_true.extend(true for (true, leaf) in zip(true, is_leaf) if leaf)

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

    def _load_backbone_weights(self, path: Path) -> None:
        """
        Load backbone weights from a saved FlatModel directory.

        Raises ValueError if the backbone architecture in the FlatModel
        does not match config.model.backbone.

        Args:
            path: Path to a FlatModel save directory containing
                  weights.pth and model_metadata.json.
        """
        with open(path / "model_metadata.json") as f:
            flat_metadata = json.load(f)

        if flat_metadata["backbone"] != self.backbone_name:
            raise ValueError(
                f"Backbone mismatch: config specifies '{self.backbone_name}' "
                f"but backbone_weights were trained with '{flat_metadata['backbone']}'."
            )

        state = torch.load(
            path / "weights.pth", map_location=self.config.metadata.device
        )
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state)

    # fit ----------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        valid_loader,
        collator: LCPNCollator,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict:
        """
        Train the model for the number of epochs specified in config.

        Resets history at the start of each call. The seed is incremented
        each epoch (cfg.train.seed + epoch) for reproducible but varied
        shuffling. If the backbone is frozen, only head parameters are passed
        to the optimizer. Defaults to Adam and CosineAnnealingLR if no
        optimizer or scheduler are provided.

        Args:
            train_loader: Training data loader.
            valid_loader: Validation data loader.
            collator:     LCPNCollator used to decode true labels during validation.
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
            params = (
                self.heads.parameters()
                if self.backbone_is_frozen()
                else self.parameters()
            )
            optimizer = torch.optim.Adam(
                params,
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

            valid_metrics = self.validate(valid_loader, collator, criterion)
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
        Save model weights, config, hierarchy, history, and metadata to disk.

        Creates a subdirectory under self.directory named after self.name,
        optionally suffixed with a timestamp. Saves:
        - weights.pth: model state dict
        - config.toml: training config
        - hierarchy.json: hierarchy definition
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
        self.hierarchy.save(save_dir / "hierarchy.json")

        with open(save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        with open(save_dir / "model_metadata.json", "w") as f:
            json.dump(self.model_metadata, f, indent=2)

        print(f"Saved to {save_dir}")
        return save_dir

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> Tuple["LCPNModel", Dict, Optional[Dict]]:
        """
        Load an LCPNModel from a saved directory.

        Reconstructs the model from config.toml, hierarchy.json, and
        model_metadata.json, loads weights from weights.pth, and restores
        history from history.json.

        Args:
            directory: Path to a directory created by save().

        Returns:
            Loaded LCPNModel instance.
        """
        load_dir = Path(directory)

        config = Config(load_dir / "config.toml")
        hierarchy = Hierarchy(load_dir / "hierarchy.json")

        with open(load_dir / "model_metadata.json") as f:
            model_metadata = json.load(f)

        model = cls(
            name=model_metadata["name"],
            directory=load_dir.parent,
            hierarchy=hierarchy,
            config=config,
        )

        with open(load_dir / "history.json") as f:
            model.history = json.load(f)

        state_dict = torch.load(
            load_dir / "weights.pth",
            map_location=config.metadata.device,
        )
        model.load_state_dict(state_dict)

        return model

    # helpers ------------------------------------------------------------------

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Return parameter counts for the backbone, heads, and total.

        Returns:
            Dictionary with keys "backbone", "heads", and "total".
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())

        return {
            "backbone": backbone_params,
            "heads": heads_params,
            "total": backbone_params + heads_params,
        }

    def __repr__(self):
        """Return a string summary of the model architecture and parameter count."""
        param_counts = self.get_num_parameters()
        return (
            f"LCPNModel(\n"
            f"  backbone={self.backbone_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_heads={len(self.heads)},\n"
            f"  parameters={param_counts['total']:,}\n"
            f")"
        )
