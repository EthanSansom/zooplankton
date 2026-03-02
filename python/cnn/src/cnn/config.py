from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import tomllib

import tomli_w
import torch


class Config:
    """Configuration manager for training experiments"""

    def __init__(self, config_path: Path):
        """Load configuration from TOML file"""
        self.config_path = Path(config_path)
        self._load_config()

        # The Config class "owns" the [metadata] table in the TOML file
        if hasattr(self, "metadata"):
            raise ValueError(
                "Configuration .toml file may not have a [metadata] table."
            )

        self.metadata = SimpleNamespace()
        self._set_device()
        self._set_timestamp()

    def _load_config(self):
        """Load TOML config file"""
        with open(self.config_path, "rb") as f:
            cfg_dict = tomllib.load(f)

        # Convert nested dicts to namespaces
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleNamespace(**value))
            else:
                setattr(self, key, value)

    def _set_device(self):
        """Auto-detect and set compute device"""
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.metadata.device = device

    def _set_timestamp(self):
        """Add experiment timestamp"""
        self.metadata.starttime = datetime.now()

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({attrs})"

    def to_dict(self) -> dict:
        """Convert config to dictionary with a TOML compatible structure"""

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

    def save(self, path: Path):
        """Save config to TOML file"""
        path = Path(path)

        if not path.parent.is_dir():
            raise FileNotFoundError(f"Can't find directory {path.parent}.")

        time = datetime.now()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamped_path = path.parent / f"{path.stem}_{timestamp}{path.suffix}"

        if timestamped_path.exists():
            raise FileExistsError(f"File already exists: {timestamped_path}.")

        with open(timestamped_path, "wb") as f:
            starttime = self.metadata.starttime
            duration = time - starttime
            self.metadata.starttime = starttime.isoformat()
            self.metadata.endtime = time.isoformat()
            self.metadata.duration = str(duration)

            tomli_w.dump(self.to_dict(), f)

        return timestamped_path
