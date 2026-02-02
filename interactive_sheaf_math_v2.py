import numpy as np
from datetime import datetime
import scipy.linalg  # New import for more stable null space

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

def save_test_results(v_dims, e_dims, energy, num_sections, null_space, L, spectrum):
    filename = "sheaf_testsv2.txt"
    with open(filename, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TEST RECORDED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vertex Dims: {v_dims} | Edge Dims: {e_dims}\n")
        f.write(f"Spectral Energy: {round(energy, 6)}\n")
        f.write(f"Dim of Global Sections (H0): {num_sections}\n")
        f.write(f"Spectrum: {np.round(spectrum, 6)}\n")
        f.write(f"Laplacian Matrix:\n{L}\n")
        if num_sections > 0:
            f.write(f"Basis for Global Sections:\n{np.round(null_space, 6)}\n")
        f.write(f"{'='*60}\n")
    print(f"✅ Results saved to {filename}")

while True:
    print("\n" + "#"*40)
    print("SHEAF ANALYSIS: ENERGY & NULL SPACE")
    print("#"*40)
    
    # ... (Same input logic for v_dims, e_dims, and maps) ...
    try:
        v_dims = [int(input(f"Dim of Vertex {i}: ")) for i in range(3)]
        e_dims = [int(input(f"Dim of Edge {i}: ")) for i in range(2)]
    except ValueError: break

    maps = {
        (0, 0): get_matrix_input(e_dims[0], v_dims[0], "V0 -> E0"),
        (0, 1): get_matrix_input(e_dims[0], v_dims[1], "V1 -> E0"),
        (1, 1): get_matrix_input(e_dims[1], v_dims[1], "V1 -> E1"),
        (1, 2): get_matrix_input(e_dims[1], v_dims[2], "V2 -> E1")
    }

    v_off = np.insert(np.cumsum(v_dims), 0, 0)
    e_off = np.insert(np.cumsum(e_dims), 0, 0)
    delta = np.zeros((sum(e_dims), sum(v_dims)))
    
    delta[e_off[0]:e_off[1], v_off[0]:v_off[1]] = maps[(0, 0)]
    delta[e_off[0]:e_off[1], v_off[1]:v_off[2]] = -maps[(0, 1)]
    delta[e_off[1]:e_off[2], v_off[1]:v_off[2]] = maps[(1, 1)]
    delta[e_off[1]:e_off[2], v_off[2]:v_off[3]] = -maps[(1, 2)]

    L = delta.T @ delta
    spec = np.linalg.eigvalsh(L)

    # --- NEW COMPUTATIONS ---
    # 1. Energy: Sum of squares of eigenvalues
    energy = np.sum(spec**2)

    # 2. Null Space (Global Sections)
    # We find vectors where L @ x approx 0. 
    # scipy.linalg.null_space uses SVD which is very reliable.
    null_space = scipy.linalg.null_space(L)
    num_sections = null_space.shape[1]

    print("\n" + "="*30)
    print(f"Laplacian Spectrum: {np.round(spec, 4)}")
    print(f"Spectral Energy: {round(energy, 4)}")
    print(f"Dimension of Global Sections (H0): {num_sections}")
    
    if num_sections > 0:
        print("\nBasis for Global Sections (each column is a section):")
        print(np.round(null_space, 4))
    else:
        print("\nNo non-trivial Global Sections found.")
    print("="*30)
    save_test_results(v_dims, e_dims, energy, num_sections, null_space, L, spec)

    if input("\nRun another? (y/n): ").lower() != 'y': break