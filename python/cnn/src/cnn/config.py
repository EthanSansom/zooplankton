import tomllib
import warnings
from pathlib import Path
from types import SimpleNamespace

import tomli_w
import torch


class Config:
    """
    Configuration manager for training experiments.

    Loads a TOML file and exposes each top-level table as a SimpleNamespace
    attribute, e.g. a [train] table is accessed as cfg.train.epochs. A
    [metadata] table is reserved for runtime use (e.g. cfg.metadata.device)
    and may not be defined in the config file.
    """

    def __init__(self, config_path: Path):
        """
        Load and validate a configuration from a TOML file.

        The [metadata] table is reserved for runtime use by Config and will
        raise a warning if found in the config file.

        Args:
            config_path: Path to a TOML configuration file.
        """
        self.config_path = Path(config_path)
        self._load_config()

        # The Config class "owns" the metadata table
        if hasattr(self, "metadata"):
            warnings.warn(
                "[metadata] table found in config file and will be ignored, "
                "this table is reserved for runtime use by Config.",
                UserWarning,
                stacklevel=2,
            )
            del self.metadata

        # The primary reason for the [metadata] table is to store the `torch.device()`
        self.metadata = SimpleNamespace()
        self._set_device()

    def __repr__(self):
        """Return a string representation of all non-private config attributes."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({attrs})"

    def to_dict(self) -> dict:
        """
        Convert config to a TOML-compatible dictionary.

        Runtime-only attributes (metadata, config_path, and private attributes)
        are excluded. torch.device values are converted to strings.

        Returns:
            Nested dict mirroring the original TOML structure.
        """

        result = {}
        for key, value in self.__dict__.items():
            # Ignore attributes used internally by this class (i.e. not configs)
            if key.startswith("_") or key == "config_path":
                continue

            if isinstance(value, SimpleNamespace):
                result[key] = {}
                for k, v in vars(value).items():
                    # Convert torch.device to string
                    if isinstance(v, torch.device):
                        result[key][k] = str(v)
                    else:
                        result[key][k] = v
            else:
                result[key] = value

        return result

    # getter / setter ----------------------------------------------------------

    def get_value(self, section: str, key: str):
        """
        Get a config value.

        Args:
            section: Top-level TOML table, e.g. "train".
            key:     Key within the section, e.g. "epochs".

        Returns:
            The current value.
        """
        return getattr(getattr(self, section), key)

    def set_value(self, section: str, key: str, value) -> None:
        """
        Set a config value. Only existing keys may be updated.

        Args:
            section: Top-level TOML table, e.g. "train".
            key:     Key within the section, e.g. "epochs".
            value:   New value.

        Raises:
            ValueError: If key does not exist in the section.
        """
        namespace = getattr(self, section)
        if not hasattr(namespace, key):
            raise ValueError(f"Key '{key}' does not exist in [{section}].")

        setattr(namespace, key, value)

    # read / write -------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save config to a TOML file.

        Only keys present in the original file are saved, but their current
        (potentially modified) values are used. Runtime-only attributes
        such as metadata are excluded.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        original = {
            k: v for k, v in self.to_dict().items() if k in self._source_config.keys()
        }
        with open(path, "wb") as f:
            tomli_w.dump(original, f)

    # initialization helpers ---------------------------------------------------

    def _load_config(self):
        """
        Load the TOML config file and set each top-level table as a
        SimpleNamespace attribute.
        """
        with open(self.config_path, "rb") as f:
            cfg_dict = tomllib.load(f)

        self._source_config = cfg_dict

        # Convert nested dicts to namespaces
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleNamespace(**value))
            else:
                setattr(self, key, value)

    def _set_device(self):
        """
        Auto-detect and set the compute device (MPS, CUDA, or CPU),
        stored as cfg.metadata.device.
        """
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.metadata.device = device
