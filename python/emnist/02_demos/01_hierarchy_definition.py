from cnn.hierarchy import Hierarchy
from pathlib import Path

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
HIERARCHIES_DIR = SCRIPT_DIR.parent / "00_hierarchies"

hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

# Demo -------------------------------------------------------------------------

print("HIERARCHY STATISTICS:")
print(f"Root node:        {hierarchy.root}")
print(f"Total nodes:      {len(hierarchy.nodes)}")
print(f"Parent nodes:     {hierarchy.n_parents}")
print(f"Leaf nodes:       {hierarchy.n_leaves}")
print(f"Number of levels: {hierarchy.max_level + 1}")
print()

print("HIERARCHY TREE:")
hierarchy.print_hierarchy()
print()

print("PATH EXAMPLES:")
test_leaves = ["Z", "o", "O", "p", "S"]
for leaf in test_leaves:
    if leaf in hierarchy.leaves:
        path = hierarchy.get_path_to_root(leaf)
        print(f"{leaf} : {' -> '.join(path)}")
