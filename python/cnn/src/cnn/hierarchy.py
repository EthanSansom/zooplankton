from pathlib import Path
import json


class Hierarchy:
    """Hierarchy loader, works for both LCPN and LCPL models."""

    def __init__(self, hierarchy_path: Path):
        """Load hierarchy from JSON file"""
        self.hierarchy_path = hierarchy_path

        # The expected JSON hierarchy format looks like:
        # - root : [child_1, child_2, ..., child_N]
        # - child_1 : [grandchild_11, grandchild_12, ..., grandchild_1K]
        with open(self.hierarchy_path, "r") as f:
            self.parent_to_children = json.load(f)

        self._create_node_info()
        self._validate()
        self._calculate_tree_info()

    def _create_node_info(self):
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

        self.child_to_parent = child_to_parent
        self.nodes = all_nodes
        self.root = root
        self.leaves = leaves
        self.parents = parents

    def _calculate_tree_info(self):
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
        """Check hierarchy is well-formed"""
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

    def get_path_to_root(self, node_name):
        """Walk up from a node to the root"""
        path = [node_name]
        current = node_name
        while current != self.root:
            current = self.child_to_parent[current]
            path.append(current)
        return list(reversed(path))

    def get_child_index(self, parent_name, child_name):
        """
        Return the index of a child relative to its parent. This is useful for
        creating an integer label in a LCPN model.
        """
        return self.parent_to_children[parent_name].index(child_name)

    def get_level_index(self, level, node_name):
        """
        Return the index of a node relative to its level. This is useful for
        creating an integer label in a LCL model.
        """
        return self.level_to_nodes[level].index(node_name)

    def get_parent_nodes(self):
        return self.parents

    def get_leaf_nodes(self):
        return self.leaves

    def get_node_level(self, node_name):
        return self.node_to_level[node_name]

    def node_is_leaf(self, node_name):
        return node_name in self.leaves

    def node_is_parent(self, node_name):
        return node_name in self.parents

    def num_children(self, parent_name):
        return len(self.parent_to_children[parent_name])

    def num_level_nodes(self, level):
        return len(self.node_to_level[level])

    # TODO: Maybe this is `print_tree` and there's also a `print_levels` and `print_leaves`
    def print_hierarchy(self, node=None, prefix="", is_last=True, show_level=True):
        """Print hierarchy as a tree (recursive)"""
        if node is None:
            node = self.root

        # Print current node
        if node == self.root:
            print(node)
        else:
            connector = "└── " if is_last else "├── "
            level = f" ({self.get_node_level(node)})" if show_level else ""
            print(prefix + connector + node + level)

        # Get children if parent node
        if node not in self.parent_to_children:
            return  # Leaf node

        children = self.parent_to_children[node]

        # Print each child
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1

            if node == self.root:
                new_prefix = ""
            else:
                extension = "    " if is_last else "│   "
                new_prefix = prefix + extension

            self.print_hierarchy(child, new_prefix, is_last_child)
