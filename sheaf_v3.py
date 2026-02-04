import numpy as np
import scipy.linalg
from datetime import datetime

def get_matrix_input(rows, cols, name):
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
    print(f"\n--- Enter values for {name} ({rows}x{cols}) ---")
    matrix = []
    for i in range(rows):
        while True:
            try:
                row_data = input(f"Row {i} (space-separated): ").split()
                if len(row_data) != cols:
                    print(f"Error: Expected {cols} numbers.")
                    continue
                matrix.append([float(x) for x in row_data])
                break
            except ValueError:
                print("Error: Enter numbers only.")
    return np.array(matrix)

def save_test_results(filename, label, n_v, edges, v_dims, e_dims, maps, energy, num_sections, null_space, L, spectrum):
    # Ensure filename ends in .txt
    if not filename.endswith(".txt"):
        filename += ".txt"
        
    with open(filename, "a") as f:
        f.write(f"\n{'#'*70}\n")
        f.write(f"LABEL: {label}\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GRAPH STRUCTURE: {n_v} vertices, edges: {edges}\n")
        f.write(f"STALK DIMENSIONS: Vertices={v_dims}, Edges={e_dims}\n")
        
        f.write("\n--- RESTRICTION MAPS ---\n")
        for key, matrix in maps.items():
            edge_idx, v_idx = key
            f.write(f"Map V{v_idx} -> E{edge_idx}:\n{matrix}\n")

        f.write("\n--- RESULTS ---\n")
        f.write(f"SPECTRAL ENERGY (Σλ²): {round(energy, 6)}\n")
        f.write(f"DIM OF GLOBAL SECTIONS (H0): {num_sections}\n")
        f.write(f"SPECTRUM: {np.round(spectrum, 6)}\n")
        f.write(f"\nLAPLACIAN MATRIX:\n{L}\n")
        
        if num_sections > 0:
            f.write(f"\nBASIS FOR GLOBAL SECTIONS (NULL SPACE):\n{np.round(null_space, 6)}\n")
        else:
            f.write("\nNO GLOBAL SECTIONS FOUND.\n")
        f.write(f"{'#'*70}\n")
    print(f"\n✅ All data saved to {filename}")

# Ask for filename once at the start
output_file = input("Enter the filename for output (e.g., results.txt): ")

while True:
    print("\n" + "█"*45)
    print("GENERAL SHEAF CALCULATOR")
    print("█"*45)
    
    test_label = input("Enter a label for this specific test: ")
    
    # 1. Graph Structure
    try:
        n_v = int(input("Number of vertices: "))
        n_e = int(input("Number of edges: "))
        edges = []
        for i in range(n_e):
            u, v = map(int, input(f"Edge {i} connects (e.g., '0 1'): ").split())
            edges.append((u, v))
    except Exception as e:
        print(f"Input error: {e}. Restarting...")
        continue

    # 2. Dimensions
    v_dims = [int(input(f"Dim of Vertex {i}: ")) for i in range(n_v)]
    e_dims = [int(input(f"Dim of Edge {i} ({edges[i][0]}-{edges[i][1]}): ")) for i in range(n_e)]

    # 3. Restriction Maps
    maps = {}
    for i, (u, v) in enumerate(edges):
        maps[(i, u)] = get_matrix_input(e_dims[i], v_dims[u], f"Map V{u} -> E{i}")
        maps[(i, v)] = get_matrix_input(e_dims[i], v_dims[v], f"Map V{v} -> E{i}")

    # 4. Assembly
    v_off = np.insert(np.cumsum(v_dims), 0, 0)
    e_off = np.insert(np.cumsum(e_dims), 0, 0)
    delta = np.zeros((sum(e_dims), sum(v_dims)))
    
    for i, (u, v) in enumerate(edges):
        delta[e_off[i]:e_off[i+1], v_off[u]:v_off[u+1]] = maps[(i, u)]
        delta[e_off[i]:e_off[i+1], v_off[v]:v_off[v+1]] = -maps[(i, v)]

    # 5. Math
    L = delta.T @ delta
    spec = np.linalg.eigvalsh(L)
    energy = np.sum(spec**2)
    null_space = scipy.linalg.null_space(L)
    num_sections = null_space.shape[1]

    # Save and Print
    save_test_results(output_file, test_label, n_v, edges, v_dims, e_dims, maps, energy, num_sections, null_space, L, spec)

    if input("\nRun another test? (y/n): ").lower() != 'y': break