import time
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import Config
from ..hierarchy import Hierarchy
from ..data import LCPNCollator
from ..utils import set_seed


class LCPNModel(nn.Module):
    """
    Local Classifier Per Parent Node (LCPN) hierarchical model.

    Uses a shared backbone (feature extractor) with multiple task-specific
    classification heads: one linear layer per parent node in the hierarchy.
    """

    # initialization -----------------------------------------------------------

    def __init__(
        self,
        hierarchy: Hierarchy,
        config: Config,
        pretrained: bool = True,
        in_chans: int = 1,
    ):
        """
        Initialize LCPN model

        Args:
            hierarchy: Hierarchy object defining the structure
            config: Config object (uses config.model.backbone)
            pretrained: Whether to use pretrained weights
            in_chans: Number of input channels (1 for grayscale)
        """
        super().__init__()

        self.hierarchy = hierarchy
        self.config = config
        self.backbone_name = config.model.backbone

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=in_chans,
        )
        self.feature_dim = self.backbone.num_features

        # Create one classification head per parent node
        self.heads = nn.ModuleDict()
        for parent_node in hierarchy.get_parent_nodes():
            n_classes = hierarchy.num_children(parent_node)
            self.heads[parent_node] = nn.Linear(self.feature_dim, n_classes)

    # foward -------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            Dictionary mapping parent node names to logits
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

    # inference ----------------------------------------------------------------

    def predict_greedy(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Greedy top-down prediction: follow argmax at each level.
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

    def predict_global(
        self,
        x: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Global optimal prediction: return the leaf with the highest joint
        probability across the full hierarchy.
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
        """Predict probabilities for all nodes (parents and leaves)"""

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
        """Sum per-parent-node loss over all active classifiers in the batch"""

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

    def evaluate(
        self,
        loader,
        collator: LCPNCollator,
        criterion: Optional[nn.Module] = None,
        desc: str = "  Eval",
        collect_predictions: bool = False,
    ) -> Tuple[Dict[str, float], Optional[List[str]], Optional[List[str]]]:
        self.eval()
        device = self.config.metadata.device
        total_loss = 0.0
        all_preds_greedy, all_preds_global, all_true = [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs = inputs.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = self.forward(inputs)
                total_loss += self.compute_loss(outputs, labels, criterion).item()

                preds_greedy, _ = self.predict_greedy(inputs, outputs=outputs)
                preds_global, _ = self.predict_global(inputs, outputs=outputs)
                true = collator.uncollate_label_leaves(labels)
                all_preds_greedy.extend(preds_greedy)
                all_preds_global.extend(preds_global)
                all_true.extend(true)

        n = len(all_preds_greedy)
        metrics = {
            "loss": total_loss / len(loader),
            "accuracy_greedy": sum(p == t for p, t in zip(all_preds_greedy, all_true))
            / n,
            "accuracy_global": sum(p == t for p, t in zip(all_preds_global, all_true))
            / n,
        }
        return (
            metrics,
            all_preds_greedy if collect_predictions else None,
            all_preds_global if collect_predictions else None,
            all_true if collect_predictions else None,
        )

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
        """Full training loop driven by self.config."""

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

        history = {"train": [], "valid": [], "duration_seconds": None}
        n_epochs = cfg.train.epochs
        start = time.time()

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            set_seed(cfg.train.seed + epoch)

            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            history["train"].append({"loss": train_loss})
            print(f"  Train: loss={train_loss:.4f}")

            valid_metrics, _, _, _ = self.evaluate(valid_loader, collator, criterion)
            history["valid"].append(valid_metrics)
            print(
                f"  Valid: loss={valid_metrics['loss']:.4f}, accuracy (greedy)={valid_metrics['accuracy_greedy']:.4f}, accuracy (global)={valid_metrics['accuracy_global']:.4f}"
            )

        history["duration_seconds"] = time.time() - start
        print(f"\nDone. ({history['duration_seconds']:.1f}s)")
        return history

    # helpers ------------------------------------------------------------------

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in model"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())

        return {
            "backbone": backbone_params,
            "heads": heads_params,
            "total": backbone_params + heads_params,
        }

    def __repr__(self):
        param_counts = self.get_num_parameters()
        return (
            f"LCPNModel(\n"
            f"  backbone={self.backbone_name},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  n_heads={len(self.heads)},\n"
            f"  parameters={param_counts['total']:,}\n"
            f")"
        )
