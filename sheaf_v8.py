import numpy as np
import scipy.linalg
from datetime import datetime

def build_restriction_map(v_set, e_set):
    v_list = sorted(list(v_set))
    e_list = sorted(list(e_set))
    if not e_list:
        return np.zeros((0, len(v_list)))
    matrix = np.zeros((len(e_list), len(v_list)))
    for i, e_elem in enumerate(e_list):
        if e_elem in v_list:
            matrix[i, v_list.index(e_elem)] = 1.0
    return matrix

def get_section_support(null_space, v_off, n_v):
    supports = []
    if null_space.shape[1] == 0:
        return supports
    for col in range(null_space.shape[1]):
        section = null_space[:, col]
        active_nodes = []
        for i in range(n_v):
            v_chunk = section[v_off[i]:v_off[i+1]]
            if not np.allclose(v_chunk, 0, atol=1e-8):
                active_nodes.append(i)
        supports.append(active_nodes)
    return supports

def format_full_output(label, n_v, edges, v_labels, e_labels, energy, beta_0, beta_1, supports, spectrum, basis):
    """Generates a standardized string for both terminal and file output."""
    res = []
    res.append(f"\n{'='*75}")
    res.append(f"LABEL: {label}")
    res.append(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    res.append(f"H0 DIM (beta_0): {beta_0} | H1 DIM (beta_1): {beta_1}")
    res.append(f"Euler Characteristic (chi): {beta_0 - beta_1}")
    
    res.append("\n--- BASIS FOR GLOBAL SECTIONS (H0) ---")
    if basis.size > 0:
        # Create row labels for clarity
        row_labels = []
        for i, labels in enumerate(v_labels):
            for label_val in sorted(list(labels)):
                row_labels.append(f"V{i}-{label_val}")
        
        header = "Row ID  | " + "  ".join([f"Sec{i}".ljust(6) for i in range(beta_0)])
        res.append(header)
        res.append("-" * len(header))
        for idx, row in enumerate(np.round(basis, 4)):
            label_str = row_labels[idx].ljust(7)
            row_vals = "  ".join([str(val).ljust(6) for val in row])
            res.append(f"{label_str} | {row_vals}")
    else:
        res.append("None (Stalk dimensions sum to 0)")

    res.append("\n--- GLOBAL SECTION SUPPORT ---")
    if not supports:
        res.append("None")
    for idx, nodes in enumerate(supports):
        res.append(f"Section {idx}: {nodes}")

    res.append("\n--- SPECTRAL DATA ---")
    res.append(f"Energy: {round(energy, 6)}")
    res.append(f"Spectrum: {np.round(spectrum, 4)}")
    res.append(f"{'='*75}\n")
    
    return "\n".join(res)

# --- Execution ---
output_file = input("Filename for research log: ")

while True:
    print("\n" + "="*50 + "\nSHEAF ANALYSIS: TERMINAL/LOG SYNC\n" + "="*50)
    test_label = input("Test label: ")
    
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        edges = [tuple(map(int, input(f"Edge {i} (u v): ").split())) for i in range(n_e)]
        v_labels = [set(input(f"V{i} labels: ").strip().split()) for i in range(n_v)]

        # Process Intersections
        e_labels = [v_labels[u].intersection(v_labels[v]) for u, v in edges]
        v_dims = [len(s) for s in v_labels]
        e_dims = [len(s) for s in e_labels]
        total_v_dim = sum(v_dims)

        if total_v_dim == 0:
            print("⚠️ Total stalk dimension is 0. Aborting calculation.")
            continue

        # Assembly
        v_off = np.insert(np.cumsum(v_dims), 0, 0)
        e_off = np.insert(np.cumsum(e_dims), 0, 0)
        delta = np.zeros((sum(e_dims), total_v_dim))

        for i, (u, v) in enumerate(edges):
            if e_dims[i] > 0:
                delta[e_off[i]:e_off[i+1], v_off[u]:v_off[u+1]] = build_restriction_map(v_labels[u], e_labels[i])
                delta[e_off[i]:e_off[i+1], v_off[v]:v_off[v+1]] = -1.0 * build_restriction_map(v_labels[v], e_labels[i])

        # Core Math
        L = delta.T @ delta
        spectrum = np.linalg.eigvalsh(L)
        energy = np.sum(spectrum**2)
        basis_H0 = scipy.linalg.null_space(delta)
        beta_0 = basis_H0.shape[1]
        beta_1 = scipy.linalg.null_space(delta.T).shape[1]
        supports = get_section_support(basis_H0, v_off, n_v)

        # Generate Uniform Output
        final_output = format_full_output(test_label, n_v, edges, v_labels, e_labels, 
                                          energy, beta_0, beta_1, supports, spectrum, basis_H0)
        
        # Print to Terminal
        print(final_output)
        
        # Save to File
        with open(output_file + ".txt" if not output_file.endswith(".txt") else output_file, "a", encoding="utf-8") as f:
            f.write(final_output)

    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    if input("\nRun another test? (y/n): ").lower() != 'y': break