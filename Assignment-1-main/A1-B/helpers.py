from IPython.display import display, SVG
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import pickle
import random
import os
import tempfile


def generate_graph(
    num_nodes: int, edge_prob: float = 0.05, weight_range: tuple[int, int] = (1, 20)
) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    """
    Creates a connected undirected graph with specified nodes and edge probabilities,
    and also generates unique coordinates for each node.

    Args:
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Probability of edge creation between two nodes.
        weight_range (tuple[int, int]): Range of weights for the edges.

    Returns:
        nx.Graph: A connected undirected graph with node coordinates.
        node_coords (dict[int, tuple[float, float]]): Coordinates for each node.
    """
    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)

    # Create a connected backbone
    for i in range(num_nodes - 1):
        weight = random.randint(*weight_range)
        G.add_edge(i, i + 1, weight=weight)

    # Add additional edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and not G.has_edge(i, j) and random.random() < edge_prob:
                weight = random.randint(*weight_range)
                G.add_edge(i, j, weight=weight)

    # Generate unique coordinates for each node
    node_coords = {}
    existing_coords = set()

    for node in G.nodes:
        while True:
            x = round(random.uniform(0, 10), 2)
            y = round(random.uniform(0, 10), 2)
            if (x, y) not in existing_coords:
                node_coords[node] = (x, y)
                existing_coords.add((x, y))
                break

    return G, node_coords


def visualize_graph(
    G: nx.Graph,
    heuristic_values: dict[int, float] = None,
    layout: str = "dot",  # Options: 'dot', 'neato', 'fdp', 'sfdp', etc.
) -> None:
    """
    Visualizes a graph using Graphviz and renders it as an SVG.

    Args:
        G (nx.Graph): The graph to visualize.
        heuristic_values (dict[int, float]): Optional heuristic values for nodes.
        layout (str): The layout algorithm for Graphviz ('dot', 'neato', 'fdp', 'sfdp').
    """
    A = to_agraph(G)

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", "")
        A.get_edge(u, v).attr["label"] = str(weight)

    if heuristic_values:
        for node, value in heuristic_values.items():
            node_obj = A.get_node(node)
            node_obj.attr["label"] = (
                f"<{node}<br/><font color='red'>{value:.6f}</font>>"
            )

    A.layout(prog=layout)

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        svg_path = tmp_file.name
        A.draw(svg_path)

    display(SVG(svg_path))
    os.remove(svg_path)


def visualize_search(
    G: nx.Graph,
    path: list[int],
    visited: set[int],
    heuristic_values: dict[int, float] = None,  # Added heuristic values parameter
    layout: str = "dot",  # Options: 'dot', 'neato', 'fdp', 'sfdp', etc.
) -> None:
    """
    Visualizes a graph highlighting a search path and visited nodes.

    Args:
        G (nx.Graph): The graph to visualize.
        path (list[int]): The list of nodes representing the search path.
        visited (set[int]): The set of nodes visited during the search.
        heuristic_values (dict[int, float]): Optional heuristic values for nodes.
        layout (str): The layout algorithm for Graphviz ('dot', 'neato', 'fdp', 'sfdp').

    Legend:
        - Nodes:
            - Light Blue Fill: Nodes that are part of the search path.
            - Double Circle: Start node.
            - Double Octagon: Goal node.
            - Orange Outline: Nodes that were visited during the search.
        - Edges:
            - Light Blue Lines: Edges that are part of the search path.

    Notes:
        - The search path highlights both the nodes and edges included in the path.
        - Visited nodes that are not part of the path are outlined in orange.
        - Start and goal nodes are uniquely shaped.
        - The graph is rendered as an SVG and displayed in the notebook.
    """
    A = to_agraph(G)

    # Highlight the nodes in the path
    for i, node in enumerate(path):
        node_obj = A.get_node(node)
        node_obj.attr["style"] = "filled"
        node_obj.attr["fillcolor"] = "lightblue"
        node_obj.attr["penwidth"] = 2
        if i == 0:
            node_obj.attr["shape"] = "doublecircle"
        elif i == len(path) - 1:
            node_obj.attr["shape"] = "doubleoctagon"

    for node in visited:
        node_obj = A.get_node(node)
        node_obj.attr["color"] = "orange"
        node_obj.attr["penwidth"] = 2

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = A.get_edge(u, v)
        edge.attr["color"] = "lightblue"
        edge.attr["penwidth"] = 2

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", "")
        A.get_edge(u, v).attr["label"] = str(weight)

    if heuristic_values:
        for node, value in heuristic_values.items():
            node_obj = A.get_node(node)
            node_obj.attr["label"] = (
                f"<{node}<br/><font color='red'>{value:.6f}</font>>"
            )

    A.layout(prog=layout)

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        svg_path = tmp_file.name
        A.draw(svg_path)

    display(SVG(svg_path))
    os.remove(svg_path)


def save_graph_to_file(G: nx.Graph, node_coords: dict, filename: str) -> None:
    """
    Serializes the graph into a pickle file.

    Args:
        G (nx.Graph): The graph to serialize.
        filename (str): The filename for the pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump((G, node_coords), f)
