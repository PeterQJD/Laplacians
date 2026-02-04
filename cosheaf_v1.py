import numpy as np
import scipy.linalg
from datetime import datetime

def build_extension_map(e_set, v_set):
    """Maps from the larger edge set to the smaller vertex set."""
    e_list = sorted(list(e_set))
    v_list = sorted(list(v_set))
    if not v_list:
        return np.zeros((0, len(e_list)))
    
    matrix = np.zeros((len(v_list), len(e_list)))
    for i, v_elem in enumerate(v_list):
        if v_elem in e_list:
            matrix[i, e_list.index(v_elem)] = 1.0
    return matrix

def get_costalk_support(null_space_H0, v_off, n_v):
    """Identifies which vertices contribute to the Global Costalks (Cokernel)."""
    supports = []
    if null_space_H0.shape[1] == 0:
        return supports
    for col in range(null_space_H0.shape[1]):
        vector = null_space_H0[:, col]
        active_nodes = []
        for i in range(n_v):
            v_chunk = vector[v_off[i]:v_off[i+1]]
            if not np.allclose(v_chunk, 0, atol=1e-8):
                active_nodes.append(i)
        supports.append(active_nodes)
    return supports

def format_cosheaf_output(label, n_v, edges, v_labels, e_labels, energy, h0_dim, h1_dim, supports, spectrum, basis_h0):
    res = []
    res.append(f"\n{'='*75}")
    res.append(f"COSHEAF ANALYSIS: {label}")
    res.append(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    res.append(f"H0 DIM (Global Costalks): {h0_dim} | H1 DIM: {h1_dim}")
    res.append(f"Euler Characteristic (chi): {h0_dim - h1_dim}")
    
    res.append("\n--- BASIS FOR GLOBAL COSTALKS (H0) ---")
    if basis_h0.size > 0:
        row_labels = []
        for i, labels in enumerate(v_labels):
            for val in sorted(list(labels)):
                row_labels.append(f"V{i}-{val}")
        
        header = "Row ID  | " + "  ".join([f"Cos{i}".ljust(6) for i in range(h0_dim)])
        res.append(header)
        res.append("-" * len(header))
        for idx, row in enumerate(np.round(basis_h0, 4)):
            label_str = row_labels[idx].ljust(7)
            row_vals = "  ".join([str(val).ljust(6) for val in row])
            res.append(f"{label_str} | {row_vals}")
    else:
        res.append("None (All vertex data is integrated by edges)")

    res.append("\n--- GLOBAL COSTALK SUPPORT (Vertices with unique data) ---")
    if not supports:
        res.append("None")
    for idx, nodes in enumerate(supports):
        res.append(f"Costalk {idx}: {nodes}")

    res.append("\n--- SPECTRAL DATA (Cosheaf Laplacian L0) ---")
    res.append(f"Energy: {round(energy, 6)}")
    res.append(f"Spectrum: {np.round(spectrum, 4)}")
    res.append(f"{'='*75}\n")
    
    return "\n".join(res)

# --- Main Execution ---
output_file = input("Filename for research log: ")

while True:
    print("\n" + "="*50 + "\nCOSHEAF ANALYSIS: UNION RULE\n" + "="*50)
    test_label = input("Test label: ")
    
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        edges = [tuple(map(int, input(f"Edge {i} (u v): ").split())) for i in range(n_e)]
        v_labels = [set(input(f"V{i} labels: ").strip().split()) for i in range(n_v)]

        # Union Rule for Edges
        e_labels = [v_labels[u].union(v_labels[v]) for u, v in edges]
        v_dims = [len(s) for s in v_labels]
        e_dims = [len(s) for s in e_labels]
        
        total_v_dim = sum(v_dims)
        total_e_dim = sum(e_dims)

        # Boundary Matrix Assembly (Rows: Vertices, Cols: Edges)
        v_off = np.insert(np.cumsum(v_dims), 0, 0)
        e_off = np.insert(np.cumsum(e_dims), 0, 0)
        boundary = np.zeros((total_v_dim, total_e_dim))

        for i, (u, v) in enumerate(edges):
            if e_dims[i] > 0:
                # Map Edge i to Vertex u (+) and Vertex v (-)
                boundary[v_off[u]:v_off[u+1], e_off[i]:e_off[i+1]] = build_extension_map(e_labels[i], v_labels[u])
                boundary[v_off[v]:v_off[v+1], e_off[i]:e_off[i+1]] = -1.0 * build_extension_map(e_labels[i], v_labels[v])

        # Core Cosheaf Math
        L0 = boundary @ boundary.T
        spectrum = np.linalg.eigvalsh(L0)
        energy = np.sum(spectrum**2)
        
        # H0 is the Cokernel of boundary, which is the Null Space of boundary.T
        basis_h0 = scipy.linalg.null_space(boundary.T)
        h0_dim = basis_h0.shape[1]
        
        # H1 is the Null Space of the boundary matrix itself
        basis_h1 = scipy.linalg.null_space(boundary)
        h1_dim = basis_h1.shape[1]
        
        supports = get_costalk_support(basis_h0, v_off, n_v)

        # Output Generation
        final_output = format_cosheaf_output(test_label, n_v, edges, v_labels, e_labels, 
                                            energy, h0_dim, h1_dim, supports, spectrum, basis_h0)
        
        print(final_output)
        with open(output_file + ".txt" if not output_file.endswith(".txt") else output_file, "a", encoding="utf-8") as f:
            f.write(final_output)

    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    if input("\nRun another cosheaf test? (y/n): ").lower() != 'y': break