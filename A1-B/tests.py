from graph import Graph

TOLERANCE = 1e-6

def test_euclidean_distance(euclidean_distance_function: callable) -> None:
    """
    Test the provided Euclidean distance function for correctness.

    This function verifies that the given Euclidean distance function
    correctly calculates the distances between pairs of nodes based on
    their coordinates. It uses a predefined set of node coordinates
    and asserts that the computed distances match the expected values.

    Parameters:
    euclidean_distance_function (callable): A function that takes two node
    identifiers and a dictionary of node coordinates, and returns the
    Euclidean distance between the two nodes.

    Raises:
    AssertionError: If any of the computed distances do not match the
    expected values.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    node_coords = {
        1: (1.24, 2.56),
        2: (2.39, 1.86),
        3: (17.82, 3.14),
        4: (2.72, 6.93),
    }

    assert (
        abs(
            euclidean_distance_function(node1=1, node2=2, node_coords=node_coords)
            - 1.3462912017836262
        )
        <= TOLERANCE
    )
    assert (
        abs(
            euclidean_distance_function(node1=1, node2=3, node_coords=node_coords)
            - 16.59014165099262
        )
        <= TOLERANCE
    )
    assert (
        abs(
            euclidean_distance_function(node1=1, node2=4, node_coords=node_coords)
            - 4.613816207869576
        )
        <= TOLERANCE
    )
    assert (
        abs(
            euclidean_distance_function(node1=2, node2=3, node_coords=node_coords)
            - 15.483000355228311
        )
        <= TOLERANCE
    )
    assert (
        abs(
            euclidean_distance_function(node1=2, node2=4, node_coords=node_coords)
            - 5.080728294250736
        )
        <= TOLERANCE
    )
    assert (
        abs(
            euclidean_distance_function(node1=3, node2=4, node_coords=node_coords)
            - 15.568368572204346
        )
        <= TOLERANCE
    )

    print("All tests passed!")


def test_greedy_best_first_search(
    greedy_search_function: callable, h_func: callable
) -> None:
    """
    Tests the implementation of the greedy best-first search algorithm.

    This function verifies that the provided greedy search function correctly finds
    the optimal path in two sample graphs. It checks that the returned paths match
    the expected paths and that the number of visited nodes is within the expected
    bounds.

    Parameters:
    greedy_search_function (callable): The greedy best-first search function to test.
    h_func (callable): A heuristic function used by the greedy search algorithm.

    Raises:
    AssertionError: If the computed paths do not match the expected paths or if the
    number of visited nodes does not meet the specified criteria.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    graph_1_path = [0, 5, 12, 13, 14]
    graph_1_vs = {0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    graph_2_path = [0, 2, 7, 8, 9]
    graph_2_vs = {0, 2, 3, 4, 7, 8, 9}

    graph_1 = Graph(filename="sample_graphs/greedy_search_1.pkl")
    student_path_1, student_vs_1 = greedy_search_function(
        graph=graph_1, start=0, goal=14, h_func=h_func
    )

    graph_2 = Graph(filename="sample_graphs/greedy_search_2.pkl")
    student_path_2, student_vs_2 = greedy_search_function(
        graph=graph_2, start=0, goal=9, h_func=h_func
    )

    # paths should match, and size of visited nodes should be greater than or equal to the path length and meet benchmark
    assert student_path_1 == graph_1_path
    assert len(student_vs_1) <= len(graph_1_vs) and len(student_vs_1) >= len(
        student_path_1
    )
    assert student_path_2 == graph_2_path
    assert len(student_vs_2) <= len(graph_2_vs) and len(student_vs_2) >= len(
        student_path_2
    )

    print("All tests passed!")


def test_ucs(ucs_function: callable) -> None:
    """
    Tests the implementation of the Uniform Cost Search (UCS) algorithm.

    This function verifies that the provided UCS function correctly finds
    the optimal path in two sample graphs. It checks that the returned paths
    match the expected paths and that the number of visited nodes is within
    the expected bounds.

    Parameters:
    ucs_function (callable): The UCS function to test.

    Raises:
    AssertionError: If the computed paths do not match the expected paths or if the
    number of visited nodes does not meet the specified criteria.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    graph_1_path = [0, 1, 2, 7, 8, 13, 14]
    graph_1_vs = {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14}
    graph_2_path = [0, 1, 13, 14, 3, 18, 19]
    graph_2_vs = {0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19}

    graph_1 = Graph(filename="sample_graphs/ucs_1.pkl")
    student_path_1, student_vs_1 = ucs_function(graph=graph_1, start=0, goal=14)

    graph_2 = Graph(filename="sample_graphs/ucs_2.pkl")
    student_path_2, student_vs_2 = ucs_function(graph=graph_2, start=0, goal=19)

    # paths should match, and size of visited nodes should be greater than or equal to the path length and meet benchmark
    assert student_path_1 == graph_1_path
    assert len(student_vs_1) <= len(graph_1_vs) and len(student_vs_1) >= len(
        student_path_1
    )
    assert student_path_2 == graph_2_path
    assert len(student_vs_2) <= len(graph_2_vs) and len(student_vs_2) >= len(
        student_path_2
    )

    print("All tests passed!")


def test_bidirectional_ucs(bidirectional_ucs_function: callable) -> None:
    # graph 1
    # Path: [0, 15, 14, 8, 11, 24]
    # Forward visited nodes: {0, 1, 2}
    # Backward visited nodes: {3, 8, 11, 14, 15, 24}

    # graph 2
    # Path: [0, 3, 7, 6, 29]
    # Forward visited nodes: {0, 1, 3, 7, 17}
    # Backward visited nodes: {5, 6, 8, 10, 25, 28, 29}

    graph_1_path = [0, 15, 14, 8, 11, 24]
    graph_1_forward_vs = {0, 1, 2}
    graph_1_backward_vs = {3, 8, 11, 14, 15, 24}

    graph_2_path = [0, 3, 7, 6, 29]
    graph_2_forward_vs = {0, 1, 3, 7, 17}
    graph_2_backward_vs = {5, 6, 8, 10, 25, 28, 29}

    graph_1 = Graph(filename="sample_graphs/bidirectional_ucs_1.pkl")
    student_path_1, student_forward_vs_1, student_backward_vs_1 = (
        bidirectional_ucs_function(graph=graph_1, start=0, goal=24)
    )

    graph_2 = Graph(filename="sample_graphs/bidirectional_ucs_2.pkl")
    student_path_2, student_forward_vs_2, student_backward_vs_2 = (
        bidirectional_ucs_function(graph=graph_2, start=0, goal=29)
    )

    # paths should match, and size of visited nodes should be greater than or equal to the path length and meet benchmarks
    assert student_path_1 == graph_1_path
    assert len(student_forward_vs_1) <= len(graph_1_forward_vs) and all(
        node in graph_1.get_nodes() for node in student_forward_vs_1
    )
    assert len(student_backward_vs_1) <= len(graph_1_backward_vs) and all(
        node in graph_1.get_nodes() for node in student_backward_vs_1
    )
    assert len(student_forward_vs_1.union(student_backward_vs_1)) >= len(graph_1_path)

    assert student_path_2 == graph_2_path
    assert len(student_forward_vs_2) <= len(graph_2_forward_vs) and all(
        node in graph_2.get_nodes() for node in student_forward_vs_2
    )
    assert len(student_backward_vs_2) <= len(graph_2_backward_vs) and all(
        node in graph_2.get_nodes() for node in student_backward_vs_2
    )
    assert len(student_forward_vs_2.union(student_backward_vs_2)) >= len(graph_2_path)

    print("All tests passed!")
