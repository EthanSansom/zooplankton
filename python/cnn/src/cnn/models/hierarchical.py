import torch
import torch.nn as nn
import timm
from typing import Dict

from ..hierarchy import Hierarchy


class LCPNModel(nn.Module):
    """
    Local Classifier Per Parent Node (LCPN) hierarchical model.

    Uses a shared backbone (feature extractor) with multiple task-specific
    classification heads: one linear layer per parent node in the hierarchy.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_chans: int = 1,
    ):
        """
        Initialize LCPN model

        Args:
            hierarchy: Hierarchy object defining the structure
            backbone: Name of backbone model (from timm)
            pretrained: Whether to use pretrained weights
            in_chans: Number of input channels (1 for grayscale)
        """
        super().__init__()

        self.hierarchy = hierarchy
        self.backbone_name = backbone

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=in_chans,
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Create one classification head per parent node
        self.heads = nn.ModuleDict()
        for parent_node in hierarchy.get_parent_nodes():
            n_classes = hierarchy.num_children(parent_node)
            self.heads[parent_node] = nn.Linear(self.feature_dim, n_classes)

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
        features = self.backbone(x)  # (batch_size, feature_dim)

        outputs = {}
        for node_name, head in self.heads.items():
            outputs[node_name] = head(features)  # (batch_size, n_classes)

        return outputs

    def predict_greedy(self, x: torch.Tensor) -> tuple:
        """
        Greedy top-down inference: follow argmax at each level

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            predictions: List of predicted leaf node names (one per image)
            paths: List of paths taken (list of node names from root to leaf)
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x)

            batch_size = x.shape[0]
            predictions = []
            paths = []

            for i in range(batch_size):
                current_node = self.hierarchy.root
                path = [current_node]

                # Iterate through each parent node on the hierarchy and use
                # the corresponding local classifier. If the `grandparent`
                # classifier emits logits = [1, 10, 2] then the `parent`
                # classifier at logits[1] == 10 is used as the next classifier.
                while current_node in self.hierarchy.parent_to_children:
                    logits = outputs[current_node][i]

                    # Predict child greedily, take the maximum logit value,
                    # no need for softmax.
                    pred_index = torch.argmax(logits).item()

                    # Get child name
                    children = self.hierarchy.parent_to_children[current_node]

                    # Check if valid child index, this shouldn't happen
                    if pred_index >= len(children):
                        raise ValueError(
                            f"Predicted index {pred_index} exceeds the number of child classes {len(children)}."
                        )

                    child_node = children[pred_index]
                    path.append(child_node)
                    current_node = child_node

                predictions.append(current_node)
                paths.append(path)

        return predictions, paths

    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict probabilities for all nodes (parents and leaves)
        """
        self.eval()

        with torch.no_grad():
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
