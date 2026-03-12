import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List

from .hierarchy import Hierarchy

# TODO:
# - LCLDataset, LCLCollator (multi classifier per level)
# - LCNDataset, LCNCollator (binary classifier per node)
#
# This is a long-term nice-to-have. You'll want to think about shared
# functionality between these approaches. For example, constructing the
# batch tensor dict in Collator __call__ is quite similar between these.
# They only differ in *which* nodes have an associated classifier.
#
# Consider a HierarchicalCollator abstract/base class, and similar for
# Flat vs. LCL vs. LCN vs. LCPN. We can abstract-away the specific node-to-child
# relationships, for example:
# - Instead of get_parent_nodes() to find LCPN classifiers, create generics:
#   - get_classifier_nodes() : Return all nodes with a classifier, root for `Flat`, parents for `LCPN`
#   - get_classifier_node_class_nodes(node) : Return the nodes *classified* by this node, leaves for `Flat`, children for `LCPN`
#   - get_classifier_node_class_node_index(classifier_node, class_node) : Think of better names, but you get the idea
#
# Potentially, put this in a `Hierarchy` wrapper. `Hierarchy` is responsible
# for managing a generic tree structure, `LCPNHierarchy` is responsible for
# LCPN-specific accessors (e.g. `get_classifier_nodes()` gets parents-nodes).

# TODO: One view that I would like is a "classifier-view" dictionary, which maps
# each classifier (head) to all of it's classes as a { label : index } dictionary.
# classifier_dict -> {
#  root : { round : 0, straight : 1 },
#  round : { O : 0, C : 1, Q : 2},
#  straight : { 1 : 0, L : 1, l : 2, I : 3 }
# }
#
# The leaf dictionary is similar, for "Q":
# leaf_dict -> {
#   root : round,
#   round : O,
#   straight : None # Or some sentinel
# }
#
# To get the indices of a leaf for the batched tensor:
#
# for leaf_classifier, leaf_class in leaf_dict:
#   classifier = classifier_dict[leaf_classifier]
#   if leaf_class is None:
#      class_index = -1 # Sentinel, this leaf has no class in the `classifer`
#   else:
#      class_index = classifier[leaf_class]
#
# Using this terminology, `leaf_index_to_name` is actually the `flat_classifier_dict`,
# with an implied `root` key: { root : { A : 0, B : 1, C : 2, ...} }.


class LCPNDataset(Dataset):
    """
    Wraps a flat dataset to produce hierarchical labels for an input dataset
    to a Local Classifier per Parent Node (LCPN) model.

    Takes a base dataset with (image, leaf_label_index) and converts to
    (image, hierarchical_labels_dict) based on hierarchy structure.
    """

    # TODO: Maybe an alternative `label_to_name` which maps strings?
    def __init__(
        self,
        base_dataset: Dataset,
        hierarchy: Hierarchy,
        leaf_index_to_name: Dict[int, str],
    ):
        """
        Args:
            base_dataset: Dataset returning (image, leaf_class_index)
            hierarchy: Hierarchy object defining the structure
            leaf_index_to_name: Mapping from leaf class indices to node names
                              e.g., {0: 'C', 1: 'O', 2: 'Q', ...}
        """
        self.base_dataset = base_dataset
        self.hierarchy = hierarchy
        self.leaf_index_to_name = leaf_index_to_name

        for leaf_name in leaf_index_to_name.values():
            if leaf_name not in hierarchy.leaves:
                raise ValueError(
                    f"Leaf '{leaf_name}' in leaf_index_to_name not found in hierarchy leaves"
                )

        self._compute_labels()

    def _compute_labels(self):
        """
        Compute hierarchical labels for each leaf class index

        Creates mapping: leaf_index -> {
            parent_node_name: child_index,
            child_node_name: grandchild_index,
            ...
        }
        """
        self.leaf_index_to_labels = {}

        for leaf_index, leaf_name in self.leaf_index_to_name.items():
            path = self.hierarchy.get_path_to_root(leaf_name)

            labels = {}
            for i in range(len(path) - 1):
                parent = path[i]
                child = path[i + 1]

                child_index = self.hierarchy.get_child_index(parent, child)
                labels[parent] = child_index

            self.leaf_index_to_labels[leaf_index] = labels

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get sample with hierarchical labels

        Args:
            index: Sample index

        Returns:
            image: Image tensor
            labels: Dict mapping parent node names to child class indices
                    Only includes nodes on the path from root to leaf
        """
        # Get image and flat label from base dataset
        image, leaf_class_index = self.base_dataset[index]

        # Get pre-computed hierarchical labels
        hierarchical_labels = self.leaf_index_to_labels[leaf_class_index]

        return image, hierarchical_labels


class LCPNCollator:
    """
    Collate function for batching LCPNDataset datasets

    Converts list of (image, labels_dict) into batched tensors
    """

    def __init__(self, hierarchy: Hierarchy):
        """
        Args:
            hierarchy: Hierarchy object to get all parent nodes
        """
        self.hierarchy = hierarchy
        self.all_parent_nodes = hierarchy.get_parent_nodes()

    def __call__(self, batch: List[Tuple[torch.Tensor, Dict[str, int]]]):
        """
        Collate batch of hierarchical samples

        Args:
            batch: List of (image, labels_dict) tuples

        Returns:
            images: Batched image tensor (batch_size, C, H, W)
            labels: Dict mapping node names to batched label tensors (batch_size,)
        """
        images = []
        node_to_child_index_dicts = []

        for image, node_to_child_index_dict in batch:
            images.append(image)
            node_to_child_index_dicts.append(node_to_child_index_dict)

        images = torch.stack(images)

        # Each LCPNDataset element has a dictionary of labels like:
        # { root : parent_idx, parent_name : child_idx, child_name : grandchild_idx, ... }
        # Where the index says "which child is this", used as the integer-form
        # of the label in a Tensor, using 0-based indexing.
        #
        # Consider the following hierarchy for a subset of EMNIST data:
        # - root : [round, long]
        # - round : [o, c, O, C]
        # - long : [l, L, i, I]
        #
        # An image "i" will have labels: { root : 1, long : 2} corresponding
        # to it's hierarchical class of [root, long, i].

        # For a Local Classifier per Parent Node (LCPN) model, we create a
        # tensor of labels for each parent-node. If this batch is {i, L}, then:
        # batch_labels = {
        #   root : tensor([1, 1]),    # Index of `long` class in `root` classifier
        #   round : tensor([-1, -1]), # Missing sentinel, since i, L aren't in `round`
        #   long : tensor([2, 1])     # Index of `i`, `L` in `long` classifier
        # }
        batch_labels = {}
        for parent_node in self.all_parent_nodes:
            child_node_indices = []

            for node_to_child_index in node_to_child_index_dicts:
                if parent_node in node_to_child_index:
                    child_node_indices.append(node_to_child_index[parent_node])
                else:
                    child_node_indices.append(-1)  # Off-path

            batch_labels[parent_node] = torch.tensor(
                child_node_indices, dtype=torch.long
            )

        return images, batch_labels

    def unbatch(self, batch_labels: Dict[str, torch.Tensor], index: int) -> List[str]:
        """
        Extract path for a single sample from batched labels

        Args:
            batch_labels: Batched labels dict from __call__
                        {node_name: tensor of shape (batch_size,)}
            index: Index of sample to extract

        Returns:
            Path from root to leaf as list of node names
            e.g., ['root', 'straight', 'angular', 'A']
        """
        # Extract sample labels (only on-path nodes)
        sample_labels = {}
        for node_name, batch_tensor in batch_labels.items():
            label_value = batch_tensor[index].item()
            if label_value != -1:  # Only include on-path nodes
                sample_labels[node_name] = label_value

        # Build path from root to leaf
        path = [self.hierarchy.root]
        current_node = self.hierarchy.root

        while current_node in self.hierarchy.parent_to_children:
            # Check if we have a label for this node
            if current_node not in sample_labels:
                break  # Reached end of path

            # Get child index and name
            child_idx = sample_labels[current_node]
            children = self.hierarchy.parent_to_children[current_node]
            child_node = children[child_idx]

            # Add to path and continue
            path.append(child_node)
            current_node = child_node

        return path

    def unbatch_all(self, batch_labels: Dict[str, torch.Tensor]) -> List[List[str]]:
        """
        Extract paths for all samples from batched labels

        Args:
            batch_labels: Batched labels dict from __call__

        Returns:
            List of paths (one per sample)
            e.g., [
                ['root', 'curved', 'open_curve', 'C'],
                ['root', 'straight', 'vertical', 'I'],
                ...
            ]
        """
        batch_size = next(iter(batch_labels.values())).shape[0]
        return [self.unbatch(batch_labels, i) for i in range(batch_size)]
