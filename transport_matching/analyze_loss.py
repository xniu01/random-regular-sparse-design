import numpy as np
import networkx as nx

# Generate scenarios for unipartite graph where each node is retained with probability p
def generate_unipartite_scenarios(num_scenario, n, p, seed=None):
    '''
    :param num_scenario: number of scenarios
    :param n: number of nodes
    :param p: probability of retaining each node
    :return: list of scenario node masks (True if node is retained, False otherwise)
    '''
    if seed is not None:
        np.random.seed(seed)
    
    scenarios = []
    for i in range(num_scenario):
        # Generate binary mask for each node
        node_mask = np.random.rand(n) < p
        scenarios.append(node_mask)
    
    return scenarios

# Generate a random d-regular unipartite graph using configuration model
def random_regular_unipartite(n, d, seed=None):
    """
    Generate a random d-regular unipartite graph using strict configuration model
    :param n: number of nodes
    :param d: degree of each node
    :param seed: random seed
    :return: adjacency matrix of the graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Check if d*n is even (necessary for perfect matching)
    if (d * n) % 2 != 0:
        raise ValueError("d*n must be even for a d-regular graph to exist")
    
    # Step 1: Create half-edges for each node
    # Each node has d half-edges
    half_edges = []
    for node in range(n):
        for _ in range(d):
            half_edges.append(node)
    
    # Step 2: Randomly pair half-edges (configuration model)
    np.random.shuffle(half_edges)
    
    # Pair consecutive half-edges
    edges = []
    for i in range(0, len(half_edges), 2):
        if i + 1 < len(half_edges):
            u = half_edges[i]
            v = half_edges[i + 1]
            edges.append((u, v))
    
    # Step 3: Create adjacency matrix (allowing self-loops and multiple edges)
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] += 1
        A[v, u] += 1  # Make symmetric
    
    # Step 4: Remove self-loops and multiple edges to get simple graph
    # Remove self-loops
    np.fill_diagonal(A, 0)
    
    # Remove multiple edges (keep only 1 if multiple edges exist)
    A = (A > 0).astype(int)
    
    # Note: After removing self-loops and multiple edges, 
    # the resulting graph may not be exactly d-regular
    # This is the expected behavior of the configuration model
    
    return A

# Generate a complete graph
def complete_graph(n):
    """
    Generate a complete graph adjacency matrix
    :param n: number of nodes
    :return: adjacency matrix of the complete graph
    """
    A = np.ones((n, n), dtype=int)
    np.fill_diagonal(A, 0)  # Remove self-loops
    return A

# Calculate maximum matching value in residual graph after removing nodes
def cost_function(graph_adj, node_mask):
    """
    Calculate maximum matching value in residual graph after removing nodes
    :param graph_adj: adjacency matrix of the graph
    :param node_mask: boolean array indicating which nodes are retained
    :param dH: high demand value (not used in this version)
    :param dL: low demand value (not used in this version)
    :return: maximum matching value
    """
    # Get indices of retained nodes
    retained_nodes = np.where(node_mask)[0]
    
    if len(retained_nodes) == 0:
        return 0
    
    # Create subgraph with only retained nodes
    residual_adj = graph_adj[np.ix_(retained_nodes, retained_nodes)]
    
    # Create NetworkX graph for residual graph
    G = nx.Graph()
    
    # Add edges (map back to original node indices)
    for i, orig_i in enumerate(retained_nodes):
        for j, orig_j in enumerate(retained_nodes):
            if i < j and residual_adj[i, j] == 1:  # Only upper triangle
                G.add_edge(orig_i, orig_j)
    
    # Find maximum matching in residual graph
    # Use max_weight_matching with maxcardinality=True for maximum matching
    matching = nx.max_weight_matching(G, maxcardinality=True)
    
    # Return the number of matched edges
    return len(matching)

# Generate an Erdős-Rényi random graph
def ER_unipartite(n, d, seed=None):
    """
    Generate an Erdős-Rényi random graph with expected degree d
    :param n: number of nodes
    :param d: expected degree (used to calculate connection probability p = d/(n-1))
    :param seed: random seed
    :return: adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate connection probability p = d/(n-1) for expected degree d
    p = d / (n - 1) if n > 1 else 0
    
    # Generate adjacency matrix
    A = np.zeros((n, n), dtype=int)
    
    # For each pair of nodes, connect with probability p
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            if np.random.rand() < p:
                A[i, j] = 1
                A[j, i] = 1  # Make symmetric
    
    return A

# Generate a d-regular k-chain unipartite graph
def k_chain_unipartite(n, d, seed=None):
    """
    Generate a d-regular k-chain unipartite graph
    :param n: number of nodes
    :param d: degree of each node (should be even and <= n-1)
    :param seed: random seed
    :return: adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    if d > n - 1:
        raise ValueError("d cannot be larger than n-1")
    if d % 2 != 0:
        raise ValueError("d must be even for k-chain")
    
    A = np.zeros((n, n), dtype=int)
    
    # For k-chain, each node connects to d/2 neighbors on each side
    half_d = d // 2
    
    for i in range(n):
        # Connect to d/2 neighbors on the right
        for j in range(1, half_d + 1):
            right_neighbor = (i + j) % n
            if right_neighbor != i:  # Avoid self-loops
                A[i, right_neighbor] = 1
                A[right_neighbor, i] = 1
        
        # Connect to d/2 neighbors on the left
        for j in range(1, half_d + 1):
            left_neighbor = (i - j) % n
            if left_neighbor != i:  # Avoid self-loops
                A[i, left_neighbor] = 1
                A[left_neighbor, i] = 1
    
    return A

if __name__ == "__main__":
    pass