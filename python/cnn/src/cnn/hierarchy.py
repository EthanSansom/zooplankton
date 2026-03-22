import json
from pathlib import Path


class Hierarchy:
    """
    Class hierarchy manager. Creates a tree-structured class hierarchy loaded
    from a JSON file specifying parent and child class relationships.

    Supports both LCPN and LCPL models. Provides traversal utilities,
    index lookups, and tree statistics.

    Expected JSON format:
        {
            "root": ["child_1", "child_2", ..., "child_N"],
            "child_1": ["grandchild_1", "grandchild_2", ..., "grandchild_K"],
            ...
        }
    """

    def __init__(self, hierarchy_path: Path):
        """
        Load and validate a hierarchy from a JSON file.

        Args:
            hierarchy_path: Path to a JSON file in parent-to-children format.
        """
        self.hierarchy_path = hierarchy_path

        # The expected JSON hierarchy format looks like:
        # - root : [child_1, child_2, ..., child_N]
        # - child_1 : [grandchild_11, grandchild_12, ..., grandchild_1K]
        with open(self.hierarchy_path, "r") as f:
            self.parent_to_children = json.load(f)

        self._create_node_info()
        self._validate()
        self._calculate_tree_info()

    # hierarchy querying -------------------------------------------------------

    def get_path_to_root(self, node_name):
        """Return the path from the root to a given node, inclusive."""
        path = [node_name]
        current = node_name
        while current != self.root:
            current = self.child_to_parent[current]
            path.append(current)
        return list(reversed(path))

    def get_child_index(self, parent_name, child_name):
        """
        Return the index of a child relative to its parent's children list.
        Useful for constructing integer labels in an LCPN model.
        """
        return self.parent_to_children[parent_name].index(child_name)

    def get_level_index(self, level, node_name):
        """
        Return the index of a node within its level.
        Useful for constructing integer labels in an LCL model.
        """
        return self.level_to_nodes[level].index(node_name)

    def get_parent_nodes(self):
        """Return all parent (non-leaf) nodes."""
        return self.parents

    def get_leaf_nodes(self):
        """Return all leaf nodes."""
        return self.leaves

    def get_node_level(self, node_name):
        """Return the depth level of a node (root is level 0)."""
        return self.node_to_level[node_name]

    def node_is_leaf(self, node_name):
        """Return True if the node is a leaf."""
        return node_name in self.leaves

    def node_is_parent(self, node_name):
        """Return True if the node has children."""
        return node_name in self.parents

    def num_children(self, parent_name):
        """Return the number of children of a parent node."""
        return len(self.parent_to_children[parent_name])

    def num_level_nodes(self, level):
        """Return the number of nodes at a given depth level."""
        return len(self.level_to_nodes[level])

    def print_hierarchy(self, node=None, show_level=True):
        """
        Print the hierarchy as a tree.

        Args:
            node:       Node to print from. Defaults to root.
            show_level: If True, print the depth level next to each node name.
        """
        if node is None:
            node = self.root
        print(node)
        self._print_hierarchy_recursive_(node, prefix="", show_level=show_level)

    def print_hierarchy_recursive_(self, node, prefix, show_level, is_last=True):
        """
        Recursive helper for print_hierarchy(). Not intended for direct use.

        Args:
            node:       Current node being printed.
            prefix:     Current indentation prefix.
            is_last:    Whether this node is the last child of its parent.
            show_level: If True, print the depth level next to each node name.
        """
        if node not in self.parent_to_children:
            return

        children = self.parent_to_children[node]
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            connector = "└── " if is_last_child else "├── "
            level = f" ({self.get_node_level(child)})" if show_level else ""
            print(prefix + connector + child + level)

            extension = "    " if is_last_child else "│   "
            self._print_hierarchy_recursive_(
                child, prefix + extension, show_level, is_last_child
            )

    # read / write -------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save hierarchy to a JSON file.

        Args:
            path: Destination file path (e.g. Path("hierarchies/morphological.json"))
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.parent_to_children, f, indent=2)

    # initialization helpers ---------------------------------------------------

    def _create_node_info(self):
        """
        Derive and store node relationships from parent_to_children.

        Sets attributes: root, leaves, parents, nodes, child_to_parent.
        """
        parent_to_children = self.parent_to_children

        # 1. Get every node on the tree
        all_nodes = set(parent_to_children.keys())
        for children in self.parent_to_children.values():
            all_nodes.update(children)

        # 2. Map every child to its parent
        child_to_parent = {}
        for parent, children in self.parent_to_children.items():
            for child in children:
                child_to_parent[child] = parent

        # 3. Find the root node (has no parents)
        roots = [node for node in all_nodes if node not in child_to_parent]
        if len(roots) != 1:
            raise ValueError("Only one node in the hiearchy may have no parents.")
        root = roots[0]

        # 4. Find leaf nodes (have no children)
        leaves = [node for node in all_nodes if node not in parent_to_children]

        # 5. Find parent nodes (non-leaves)
        parents = list(parent_to_children.keys())

        self.nodes = all_nodes
        self.root = root
        self.leaves = leaves
        self.parents = parents
        self.nodes = all_nodes
        self.child_to_parent = child_to_parent

    def _calculate_tree_info(self):
        """
        Compute and store tree statistics and level mappings.

        Sets attributes: n_leaves, n_parents, node_to_level, level_to_nodes, max_level.
        """

        # 1. Simple statistics
        self.n_leaves = len(self.leaves)
        self.n_parents = len(self.parents)

        # 2. Map every node to it's level on the hierarchy and map every level
        #    to it's set of nodes (useful for a LCL classifier).
        self.node_to_level = {}
        self.node_to_level[self.root] = 0

        for node in self.nodes:
            if node == self.root:
                continue

            current = node
            level = 0
            while current != self.root:
                current = self.child_to_parent[current]
                level += 1

            self.node_to_level[node] = level

        self.level_to_nodes = {}
        for node, level in self.node_to_level.items():
            if level not in self.level_to_nodes:
                self.level_to_nodes[level] = []
            self.level_to_nodes[level].append(node)

        self.max_level = max(self.level_to_nodes.keys())

    def _validate(self):
        """
        Verify the hierarchy is well-formed.

        Checks for cycles and that all referenced children exist as nodes.
        Raises ValueError if either condition is violated.
        """

        # Check for cycles
        for leaf in self.leaves:
            current = leaf
            path = set()
            while current != self.root:
                if current in path:
                    raise ValueError(f"Cycle detected involving node '{current}'")
                path.add(current)
                current = self.child_to_parent[current]

        # Check all children exist as nodes
        for parent, children in self.parent_to_children.items():
            for child in children:
                if child not in self.nodes:
                    raise ValueError(f"Child '{child}' of '{parent}' not in hierarchy")
