from pathlib import Path

from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset
from cnn.hierarchy import Hierarchy

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"

flat_cfg = Config(BASE_DIR / "00_configs" / "flat_2026_03_23.toml")
lcpn_cfg = Config(BASE_DIR / "00_configs" / "lcpn_2026_03_23.toml")
hierarchy = Hierarchy(HIERARCHIES_DIR / "taxonomic_2026_01_26.json")

# Class mappings ---------------------------------------------------------------

# fmt: off
flat_class_to_index = {
    "bosminidae": 0, "eubosmina": 0,
    "daphnia": 1,
    "rotifer": 2, "trichocerca": 2, "conochilus": 2, "kellicottia": 2,
    "nauplius_copepod": 3, "cyclopoid": 4, "harpacticoid": 5, "calanoid": 6,
    "exoskeleton": 7, "fiber_hairlike": 8, "fiber_squiggly": 8, "plant_matter": 9,
    "bubbles": 10,
}

flat_index_to_name = {
    0: "bosmina", 1: "daphnia", 2: "rotifer", 3: "nauplius",
    4: "cyclopoid", 5: "harpacticoid", 6: "calanoid", 7: "exoskeleton",
    8: "fiber", 9: "plant_matter", 10: "bubbles",
}

lcpn_class_to_index = {
    "cladocera": 0,
    "bosminidae": 1, "eubosmina": 1,
    "daphnia": 2,
    "rotifer": 3, "trichocerca": 3, "conochilus": 3, "kellicottia": 3,
    "copepoda": 4,
    "nauplius_copepod": 5, "cyclopoid": 6, "harpacticoid": 7, "calanoid": 8,
    "exoskeleton": 9, "fiber_hairlike": 10, "fiber_squiggly": 10, "plant_matter": 11,
    "bubbles": 12,
}

lcpn_node_index_to_name = {
    0: "cladocera", 4: "copepoda",
    1: "bosmina", 2: "daphnia", 3: "rotifer",
    5: "nauplius", 6: "cyclopoid", 7: "harpacticoid", 8: "calanoid",
    9: "exoskeleton", 10: "fiber", 11: "plant_matter", 12: "bubbles",
}
# fmt: on

# Datasets ---------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

flat_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=flat_class_to_index,
    class_to_nmax=flat_cfg.data.class_nmax,
)

lcpn_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=lcpn_class_to_index,
    class_to_nmax=lcpn_cfg.data.class_nmax,
)

# Print ------------------------------------------------------------------------

print("Flat Dataset:")
print(flat_dataset)
print()
flat_dataset.print_classes(flat_index_to_name)

print("\nLCPN Dataset:")
print(lcpn_dataset)
print()
lcpn_dataset.print_classes(lcpn_node_index_to_name)
