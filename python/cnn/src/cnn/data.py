from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset

from .hierarchy import Hierarchy

# ImageDataset -----------------------------------------------------------------


class ImageDataset(Dataset):
    """
    Dataset for loading .tif images from a flat directory structure:

        root/
            class_a/image1.tif
            class_a/image2.tif
            class_b/image1.tif
            ...

    Note that images are converted to greyscale on load in __getitem__.

    Args:
        root:           Path to the root directory containing class subdirectories.
        transform:      Optional transforms applied to each image.
        class_to_index: Optional mapping of class name to integer label. If None,
                        classes are assigned indices alphabetically.
        class_to_nmax:  Optional cap on the number of images per class. Accepts
                        a single int (applied to all classes) or a dict mapping
                        class names to individual caps.
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        class_to_index: Optional[Dict[str, int]] = None,
        class_to_nmax: Optional[Union[Dict[str, int], int]] = None,
    ):
        """Build the dataset by scanning root for class subdirectories."""

        self.root = Path(root)
        self.transform = transform

        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())

        if class_to_index is None:
            self.class_to_index = {d.name: i for i, d in enumerate(class_dirs)}
        else:
            self.class_to_index = class_to_index

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        for class_dir in class_dirs:
            if class_dir.name not in self.class_to_index:
                continue

            label = self.class_to_index[class_dir.name]
            image_paths = sorted(class_dir.glob("*.tif"))

            if isinstance(class_to_nmax, int):
                image_paths = image_paths[:class_to_nmax]
            elif isinstance(class_to_nmax, dict) and class_dir.name in class_to_nmax:
                image_paths = image_paths[: class_to_nmax[class_dir.name]]

            for image_path in image_paths:
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image tensor, integer class label).
        """

        # `.convert("L")` converts to greyscale, which is appropriate for Zooplankton:
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        image = Image.open(self.image_paths[index]).convert("L")
        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

    def __repr__(self) -> str:
        """Return a string summary of the dataset."""
        return (
            f"ImageDataset(\n"
            f"  root={self.root},\n"
            f"  n_samples={len(self)},\n"
            f"  n_classes={len(self.class_to_index)},\n"
            f")"
        )


# LCPNDataset ------------------------------------------------------------------


class LCPNDataset(Dataset):
    """
    Wraps a flat ImageDataset to produce hierarchical labels for a
    Local Classifier per Parent Node (LCPN) model.

    Takes a base dataset with (image, class_index) and converts to
    (image, hierarchical_labels_dict) based on the hierarchy structure.
    Supports partially-labelled samples: node_index_to_name may map to
    any node in the hierarchy, not just leaves.

    Args:
        base_dataset:       Dataset returning (image, class_index).
        hierarchy:          Hierarchy object defining the tree structure.
        node_index_to_name: Mapping from class indices to node names.
                            May include both leaf nodes and internal nodes
                            for partially-labelled samples,
                            e.g. {0: 'copepod', 1: 'calanoid', 2: 'detritus'}.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        hierarchy: Hierarchy,
        node_index_to_name: Dict[int, str],
    ):
        """
        Validate node names against the hierarchy and precompute hierarchical labels.
        """
        self.base_dataset = base_dataset
        self.hierarchy = hierarchy
        self.node_index_to_name = node_index_to_name

        for node_name in node_index_to_name.values():
            if node_name not in hierarchy.nodes:
                raise ValueError(
                    f"Node '{node_name}' in `node_index_to_name` not found in `hierarchy.nodes`."
                )

        self._compute_labels()

    def _compute_labels(self):
        """
        Precompute hierarchical labels for each class index.

        For each node index, walks the path from root to node and records
        the child index at each parent, producing a dict of the form:
            { parent_name: child_index, child_name: grandchild_index, ... }

        For partially-labelled samples, the path terminates at the labelled
        node rather than a leaf.

        Sets attribute: node_index_to_labels.
        """
        self.node_index_to_labels = {}

        for node_index, node_name in self.node_index_to_name.items():
            path = self.hierarchy.get_path_to_root(node_name)

            labels = {}
            for i in range(len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                child_index = self.hierarchy.get_child_index(parent, child)
                labels[parent] = child_index

            self.node_index_to_labels[node_index] = labels

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Return a single sample with hierarchical labels.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image tensor, labels dict), where labels maps each
            on-path parent node name to the index of the correct child.
            For partially-labelled samples, only on-path nodes up to the
            labelled node are included.
        """
        # Get image and flat label from base dataset
        image, node_class_index = self.base_dataset[index]

        hierarchical_labels = self.node_index_to_labels[node_class_index]
        return image, hierarchical_labels


class LCPNCollator:
    """
    Collate function for batching LCPNDataset samples.

    Converts a list of (image, labels_dict) tuples into a batched image
    tensor and a dict of per-parent label tensors, using -1 as a sentinel
    for samples that do not pass through a given parent node. Supports
    partially-labelled samples, where the label path terminates above a leaf.

    Args:
        hierarchy: Hierarchy object used to enumerate all parent nodes.
    """

    def __init__(self, hierarchy: Hierarchy):
        """
        Args:
            hierarchy: Hierarchy object used to enumerate all parent nodes.
        """
        self.hierarchy = hierarchy
        self.all_parent_nodes = hierarchy.get_parent_nodes()

    def __call__(self, batch: List[Tuple[torch.Tensor, Dict[str, int]]]):
        """
        Collate a batch of hierarchical samples.

        Args:
            batch: List of (image, labels_dict) tuples from LCPNDataset.

        Returns:
            images:       Batched image tensor of shape (batch_size, C, H, W).
            batch_labels: Dict mapping each parent node name to a label
                            tensor of shape (batch_size,), with -1 for
                            off-path samples. For partially-labelled samples,
                            nodes below the labelled node are also -1.
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
        # - root:  [round, long]
        # - round: [o, c, O, C]
        # - long:  [l, L, i, I]
        #
        # An image "i" will have labels: { root : 1, long : 2} corresponding
        # to it's hierarchical class of [root, long, i].

        # For a Local Classifier per Parent Node (LCPN) model, we create a
        # tensor of labels for each parent-node. Consider a batch containing
        # samples "i" and "L", then:
        # batch_labels = {
        #   root:  tensor([1, 1]),   # Index of `long` class in `root` classifier
        #   round: tensor([-1, -1]), # Missing sentinel, since `i`, `L` aren't in `round`
        #   long:  tensor([2, 1])    # Index of `i`, `L` in `long` classifier
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

    def uncollate_label_path(
        self, batch_labels: Dict[str, torch.Tensor], index: int
    ) -> List[str]:
        """
        Extract the root-to-node path for a single sample from batched labels.

        For partially-labelled samples, the path terminates at the labelled
        node rather than a leaf.

        Args:
            batch_labels: Batched labels dict from __call__,
                          mapping node names to tensors of shape (batch_size,).
            index:        Index of the sample to extract.

        Returns:
            Path from root to labelled node as a list of node names,
            e.g. ['root', 'straight', 'vertical', 'i'].
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

    def uncollate_label_leaf(
        self, batch_labels: Dict[str, torch.Tensor], index: int
    ) -> str:
        """
        Extract the deepest labelled node for a single sample from batched labels.

        For fully-labelled samples this is a leaf node; for partially-labelled
        samples this is the deepest on-path node.

        Args:
            batch_labels: Batched labels dict from __call__,
                          mapping node names to tensors of shape (batch_size,).
            index:        Index of the sample to extract.

        Returns:
            Name of the deepest labelled node for the sample.
        """
        return self.uncollate_label_path(batch_labels, index)[-1]

    def uncollate_label_paths(
        self, batch_labels: Dict[str, torch.Tensor]
    ) -> List[List[str]]:
        """
        Extract root-to-node paths for all samples from batched labels.

        For partially-labelled samples, paths terminate at the labelled node,
        e.g. a sample labelled "i" returns ['root', 'straight', 'vertical', 'i']
        while a sample labelled "straight" returns ['root', 'straight'].

        Args:
            batch_labels: Batched labels dict from __call__,
                          mapping node names to tensors of shape (batch_size,).

        Returns:
            List of paths, one per sample,
            e.g. [['root', 'straight', 'vertical', 'i'], ['root', 'mixed'], ...].
        """
        batch_size = next(iter(batch_labels.values())).shape[0]
        return [self.uncollate_label_path(batch_labels, i) for i in range(batch_size)]

    def uncollate_label_leaves(
        self, batch_labels: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], List[bool]]:
        """
        Extract the deepest labelled node for all samples from batched labels.

        Args:
            batch_labels: Batched labels dict from __call__,
                          mapping node names to tensors of shape (batch_size,).

        Returns:
            leaves:  Deepest labelled node per sample (leaf node name for
                     fully-labelled samples, internal node name for partial).
            is_leaf: Boolean per sample, True if the label is a leaf node.
        """
        paths = self.uncollate_label_paths(batch_labels)
        leaves = [path[-1] for path in paths]
        is_leaf = [self.hierarchy.node_is_leaf(leaf) for leaf in leaves]
        return leaves, is_leaf
