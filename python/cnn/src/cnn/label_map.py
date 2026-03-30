import json
from pathlib import Path
from typing import Dict, List


class LabelMap:
    """
    Maps human-readable label names to the directories that contain their
    images, and derives the class_to_index dict used by ImageDataset.

    Expected JSON format:
        {
            "class1": ["directory11", "directory12"],
            "class2": ["directory21"],
            "class3": ["directory31", "directory32", "directory33"]
        }
    """

    def __init__(self, label_map_path: Path):
        """
        Load and validate a label mapping from a JSON file.

        Args:
            label_map_path: Path to a JSON file containing a label to directories mapping.
        """
        self.label_map_path = Path(label_map_path)

        with open(self.label_map_path, "r") as f:
            self.label_to_dirs: Dict[str, List[str]] = json.load(f)

        self._validate()
        self._build_mappings()

    # helpers ------------------------------------------------------------------

    def labels(self) -> List[str]:
        """Ordered list of label names (matches index order)."""
        return list(self.label_to_dirs.keys())

    def n_classes(self) -> int:
        """Number of distinct labels (i.e. number of classes)."""
        return len(self.label_to_dirs)

    def __repr__(self) -> str:
        max_label_len = max(len(label) for label in self.label_to_dirs)
        lines = ["LabelMap("]
        for label, dirs in self.label_to_dirs.items():
            idx = self.label_to_index[label]
            dirs_str = ", ".join(dirs)
            lines.append(
                f"  {idx:>3}  {label + ':':<{max_label_len + 1}}  [{dirs_str}]"
            )
        lines.append(")")
        return "\n".join(lines)

    # initialisation  ----------------------------------------------------------

    def _validate(self) -> None:
        """Raise ValueError for duplicate directories across labels."""
        seen = {}
        for label, dirs in self.label_to_dirs.items():
            if not isinstance(dirs, list) or len(dirs) == 0:
                raise ValueError(
                    f"Label '{label}' must map to a non-empty list of directories."
                )
            for dir in dirs:
                if dir in seen:
                    raise ValueError(
                        f"Directory '{dir}' appears under both '{seen[dir]}' and '{label}'."
                    )
                seen[dir] = label

    def _build_mappings(self) -> None:
        """Derive class_to_index, label_to_index, and index_to_label."""
        self.class_to_index: Dict[str, int] = {}
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}

        for index, (label, dirs) in enumerate(self.label_to_dirs.items()):
            self.label_to_index[label] = index
            self.index_to_label[index] = label
            for directory in dirs:
                self.class_to_index[directory] = index
