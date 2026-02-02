import numpy as np

def compute_sheaf_laplacian(v_dims, edges, e_dims, restriction_maps):
    # Calculate starting positions (offsets) for each vertex/edge in the matrix
    v_offsets = np.insert(np.cumsum(v_dims), 0, 0)
    e_offsets = np.insert(np.cumsum(e_dims), 0, 0)
    
    total_v_dim = sum(v_dims)
    total_e_dim = sum(e_dims)
    
    # Create the Coboundary matrix (Delta)
    delta = np.zeros((total_e_dim, total_v_dim))
    
    for i, (u, v) in enumerate(edges):
        r_start, r_end = e_offsets[i], e_offsets[i+1]
        c_u_start, c_u_end = v_offsets[u], v_offsets[u+1]
        c_v_start, c_v_end = v_offsets[v], v_offsets[v+1]
        
        # Pull maps from our dictionary
        delta[r_start:r_end, c_u_start:c_u_end] = restriction_maps[(i, u)]
        delta[r_start:r_end, c_v_start:c_v_end] = -restriction_maps[(i, v)]
        
    # Laplacian formula: L = Delta^T * Delta
    L = delta.T @ delta
    return L

# --- DATA ENTRY SECTION ---

# 1. Define dimensions (e.g., Vertex 0 is 2D, Vertex 1 is 3D)
v_dimensions = [2, 3] 
e_dimensions = [2]    # Edge 0 is 2D
edge_list = [(0, 1)]  # One edge connecting vertex 0 to 1

# 2. Define the restriction maps (Matrices)
# Maps must be: (Edge Dimension) x (Vertex Dimension)
maps = {
    (0, 0): np.eye(2),               # 2x2 Identity matrix
    (0, 1): np.array([[1, 0, 1],     # 2x3 matrix
                      [0, 1, 1]])
}

# --- COMPUTATION ---
laplacian = compute_sheaf_laplacian(v_dimensions, edge_list, e_dimensions, maps)
spectrum = np.linalg.eigvalsh(laplacian)

print("--- Sheaf Laplacian Matrix ---")
print(laplacian)
print("\n--- Spectrum (Eigenvalues) ---")
print(np.round(spectrum, 4))