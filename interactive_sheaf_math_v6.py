import numpy as np
import scipy.linalg
from datetime import datetime

def build_restriction_map(v_set, e_set):
    v_list = sorted(list(v_set))
    e_list = sorted(list(e_set))
    # If the edge is empty, the map is effectively a 0-row matrix
    if not e_list:
        return np.zeros((0, len(v_list)))
    
    matrix = np.zeros((len(e_list), len(v_list)))
    for i, e_elem in enumerate(e_list):
        if e_elem in v_list:
            matrix[i, v_list.index(e_elem)] = 1.0
    return matrix

def save_test_results(filename, label, n_v, edges, v_labels, e_labels, energy, num_sections, null_space, L, spectrum):
    if not filename.endswith(".txt"): filename += ".txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"LABEL: {label}\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GRAPH: {n_v} vertices, Edges: {edges}\n")
        
        f.write("\n--- SYMBOLIC DATA ---\n")
        for i, v_s in enumerate(v_labels):
            v_str = " ".join(sorted(list(v_s))) if v_s else "EMPTY"
            f.write(f"V{i}: {{{v_str}}} (Dim {len(v_s)})\n")
        
        f.write("\n--- CALCULATED RESULTS ---\n")
        if L.size == 0:
            f.write("TOTAL SHEAF DIMENSION IS 0 (All stalks empty).\n")
        else:
            f.write(f"SPECTRAL ENERGY: {round(energy, 6)}\n")
            f.write(f"DIM OF GLOBAL SECTIONS (H0): {num_sections}\n")
            f.write(f"SPECTRUM: {np.round(spectrum, 6)}\n")
            f.write(f"\nLAPLACIAN MATRIX:\n{L}\n")
            if num_sections > 0:
                f.write(f"\nBASIS FOR GLOBAL SECTIONS:\n{np.round(null_space, 6)}\n")
        f.write(f"{'='*70}\n")

output_file = input("Filename for output: ")

while True:
    print("\n" + "="*45 + "\nSET-SHEAF CALCULATOR (SAFE VERSION)\n" + "="*45)
    test_label = input("Test label: ")
    
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        edges = [tuple(map(int, input(f"Edge {i} (u v): ").split())) for i in range(n_e)]

        print("\nEnter labels (e.g., 'a b'). Leave blank for empty set:")
        v_labels = []
        for i in range(n_v):
            raw = input(f"V{i} labels: ").strip().lower()
            v_labels.append(set() if raw in ['', 'empty', 'none', '{}'] else set(raw.split()))

        # Assembly
        v_dims = [len(s) for s in v_labels]
        total_v_dim = sum(v_dims)
        
        if total_v_dim == 0:
            print("\n⚠️ Total sheaf dimension is 0. No math to perform!")
            save_test_results(output_file, test_label, n_v, edges, v_labels, [], 0, 0, np.array([]), np.array([]), np.array([]))
            continue

        e_labels = [v_labels[u].intersection(v_labels[v]) for u, v in edges]
        e_dims = [len(s) for s in e_labels]
        
        delta = np.zeros((sum(e_dims), total_v_dim))
        v_off = np.insert(np.cumsum(v_dims), 0, 0)
        e_off = np.insert(np.cumsum(e_dims), 0, 0)

        for i, (u, v) in enumerate(edges):
            if e_dims[i] > 0:
                delta[e_off[i]:e_off[i+1], v_off[u]:v_off[u+1]] = build_restriction_map(v_labels[u], e_labels[i])
                delta[e_off[i]:e_off[i+1], v_off[v]:v_off[v+1]] = -1.0 * build_restriction_map(v_labels[v], e_labels[i])

        # Math logic
        L = delta.T @ delta
        spec = np.linalg.eigvalsh(L)
        energy = np.sum(spec**2)
        null_space = scipy.linalg.null_space(L)
        
        print(f"\nEnergy: {round(energy, 4)} | H0 Dimension: {null_space.shape[1]}")
        save_test_results(output_file, test_label, n_v, edges, v_labels, e_labels, 
                          energy, null_space.shape[1], null_space, L, spec)

    except Exception as e:
        print(f"Error during calculation: {e}")
    
    if input("\nRun another? (y/n): ").lower() != 'y': break