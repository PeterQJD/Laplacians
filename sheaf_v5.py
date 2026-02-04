import numpy as np
import scipy.linalg
from datetime import datetime
import sys
import io

# Ensure stdout/stderr use UTF-8 so printing Unicode (e.g. emojis, block chars)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def build_restriction_map(v_set, e_set):
    # Sort to ensure the matrix indices are predictable (Alphabetical)
    v_list = sorted(list(v_set))
    e_list = sorted(list(e_set))
    matrix = np.zeros((len(e_list), len(v_list)))
    for i, e_elem in enumerate(e_list):
        if e_elem in v_list:
            matrix[i, v_list.index(e_elem)] = 1.0
    return matrix

def save_test_results(filename, label, n_v, edges, v_labels, e_labels, energy, num_sections, null_space, L, spectrum):
    if not filename.endswith(".txt"): filename += ".txt"
    with open(filename, "a", encoding="utf-8") as f: # Added utf-8 encoding for safety
        f.write(f"\n{'='*70}\n")
        f.write(f"LABEL: {label}\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GRAPH: {n_v} vertices, Edges: {edges}\n")
        
        # Log the Symbolic Sets
        f.write("\n--- SYMBOLIC DATA ---\n")
        for i, v_s in enumerate(v_labels):
            f.write(f"Vertex {i} Set: {sorted(list(v_s))} (Dim {len(v_s)})\n")
        for i, e_s in enumerate(e_labels):
            f.write(f"Edge {i} ({edges[i]}) Intersection: {sorted(list(e_s))} (Dim {len(e_s)})\n")

        f.write("\n--- CALCULATED RESULTS ---\n")
        # use ASCII-only labels to avoid Windows 'charmap' encoding errors
        f.write(f"SPECTRAL ENERGY (sum(lambda^2)): {round(energy, 6)}\n")
        f.write(f"DIM OF GLOBAL SECTIONS (H0): {num_sections}\n")
        f.write(f"SPECTRUM: {np.round(spectrum, 6)}\n")
        f.write(f"\nLAPLACIAN MATRIX:\n{L}\n")
        
        if num_sections > 0:
            f.write(f"\nBASIS FOR GLOBAL SECTIONS:\n{np.round(null_space, 6)}\n")
        f.write(f"{'#'*70}\n")
    # avoid emoji which can trigger the console 'charmap' codec error
    try:
        print(f"Data logged to {filename}")
    except Exception:
        # fallback safe ASCII message
        print("Data logged.")

# --- Execution ---
output_file = input("Filename for output (e.g., set_tests.txt): ")

while True:
    print("\n" + "="*45)
    print("SET-INTERSECTION SHEAF CALCULATOR")
    print("="*45)
    
    test_label = input("Test label: ")
    
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        edges = [tuple(map(int, input(f"Edge {i} (u v): ").split())) for i in range(n_e)]

        print("\nEnter labels (e.g., 'a b'):")
        v_labels = [set(input(f"V{i} labels: ").split()) for i in range(n_v)]
        
        # Automatic Setup
        e_labels = []
        maps = {}
        v_dims = [len(s) for s in v_labels]
        e_dims = []

        for i, (u, v) in enumerate(edges):
            inter = v_labels[u].intersection(v_labels[v])
            e_labels.append(inter)
            e_dims.append(len(inter))
            maps[(i, u)] = build_restriction_map(v_labels[u], inter)
            maps[(i, v)] = build_restriction_map(v_labels[v], inter)

        # Matrix Assembly
        v_off = np.insert(np.cumsum(v_dims), 0, 0)
        e_off = np.insert(np.cumsum(e_dims), 0, 0)
        delta = np.zeros((sum(e_dims), sum(v_dims)))
        
        for i, (u, v) in enumerate(edges):
            delta[e_off[i]:e_off[i+1], v_off[u]:v_off[u+1]] = maps[(i, u)]
            delta[e_off[i]:e_off[i+1], v_off[v]:v_off[v+1]] = -1.0 * maps[(i, v)]

        L = delta.T @ delta
        spec = np.linalg.eigvalsh(L)
        energy = np.sum(spec**2)
        null_space = scipy.linalg.null_space(L)
        
        # Display Summary
        print(f"\nEnergy: {round(energy, 4)} | Global Sections: {null_space.shape[1]}")
        
        # Log to File
        save_test_results(output_file, test_label, n_v, edges, v_labels, e_labels, 
                          energy, null_space.shape[1], null_space, L, spec)

    except Exception as e:
        print(f"Error: {e}")
    
    if input("\nRun another? (y/n): ").lower() != 'y': break