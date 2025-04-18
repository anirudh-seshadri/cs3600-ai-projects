import pickle


class Graph:
    def __init__(self, filename: str) -> None:
        """
        Initialize the Graph by loading a graph and node coordinates from a pickle file.

        Args:
            filename (str): The path to the pickle file to load the graph and coordinates.
        """
        if not filename.endswith(".pkl"):
            raise ValueError("The file must be a pickle file.")

        with open(filename, "rb") as f:
            graph, node_coords = pickle.load(f)

        self.graph = graph
        self.node_coords = node_coords

    def get_nodes(self) -> list:
        """Return the list of nodes in the graph."""
        return list(self.graph.nodes)

    def get_edges(self) -> list:
        """Return the list of edges along with their weights as tuples (node1, node2, weight)."""
        return [(u, v, d["weight"]) for u, v, d in self.graph.edges(data=True)]

    def get_successors_and_costs(self, node: int) -> dict[int, float]:
        """Return a dictionary of successors for a node with their traversal costs."""
        return {
            neighbor: self.graph[node][neighbor]["weight"]
            for neighbor in self.graph.neighbors(node)
        }

    def get_successors(self, node: int) -> list[int]:
        """Return a list of successors for a node."""
        return list(self.graph.neighbors(node))
