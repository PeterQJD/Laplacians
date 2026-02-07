import numpy as np
import scipy.linalg
from datetime import datetime
import os
import sys

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
        
        for i, e_s in enumerate(e_labels):
            u, v = edges[i]
            e_str = " ".join(sorted(list(e_s))) if e_s else "EMPTY"
            f.write(f"E{i} ({u}-{v}): {{{e_str}}} (Dim {len(e_s)})\n")
        
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
    print("\n" + "="*45 + "\nSET-SHEAF CALCULATOR (FLEXIBLE EDGES)\n" + "="*45)
    test_label = input("Test label: ")
    
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        
        edges = []
        for i in range(n_e):
            u, v = map(int, input(f"Edge {i} (u v): ").split())
            edges.append((u, v))

        print("\nEnter vertex labels (space separated):")
        v_labels = []
        for i in range(n_v):
            raw = input(f"V{i} labels: ").strip().lower()
            v_labels.append(set() if raw in ['', 'empty', 'none', '{}'] else set(raw.split()))

        print("\nEnter edge labels (space separated).")
        print("Note: Elements should exist in both connected vertices for valid restriction maps.")
        e_labels = []
        for i in range(n_e):
            u, v = edges[i]
            raw = input(f"E{i} ({u}-{v}) labels: ").strip().lower()
            e_set = set() if raw in ['', 'empty', 'none', '{}'] else set(raw.split())
            
            # Validation check
            invalid = [item for item in e_set if item not in v_labels[u] or item not in v_labels[v]]
            if invalid:
                print(f"  ⚠️ Warning: {invalid} not found in vertex {u} or {v}. Logic may be inconsistent.")
            
            e_labels.append(e_set)

        # Assembly logic
        v_dims = [len(s) for s in v_labels]
        e_dims = [len(s) for s in e_labels]
        total_v_dim = sum(v_dims)
        
        if total_v_dim == 0:
            print("\n⚠️ Total sheaf dimension is 0. No math to perform!")
            save_test_results(output_file, test_label, n_v, edges, v_labels, e_labels, 0, 0, np.array([]), np.array([]), np.array([]))
            continue

        delta = np.zeros((sum(e_dims), total_v_dim))
        v_off = np.insert(np.cumsum(v_dims), 0, 0)
        e_off = np.insert(np.cumsum(e_dims), 0, 0)

        for i, (u, v) in enumerate(edges):
            if e_dims[i] > 0:
                # Map from u to edge
                delta[e_off[i]:e_off[i+1], v_off[u]:v_off[u+1]] = build_restriction_map(v_labels[u], e_labels[i])
                # Map from v to edge (with orientation sign -1)
                delta[e_off[i]:e_off[i+1], v_off[v]:v_off[v+1]] = -1.0 * build_restriction_map(v_labels[v], e_labels[i])

        # Laplacian and Spectral Analysis
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