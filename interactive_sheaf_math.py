import numpy as np
from datetime import datetime

def get_matrix_input(rows, cols, name):
    # Handle the "Zero Dimension" case
    if rows == 0 or cols == 0:
        print(f"--- {name} is an empty map ({rows}x{cols}) ---")
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

def save_test_results(v_dims, e_dims, maps, L, spectrum):
    filename = "sheaf_tests.txt"
    with open(filename, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"TEST RECORDED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vertex Dims: {v_dims} | Edge Dims: {e_dims}\n")
        f.write(f"Laplacian Matrix:\n{L}\n")
        f.write(f"Spectrum: {np.round(spectrum, 6)}\n")
    print(f"✅ Results appended to {filename}")

while True:
    print("\n" + "#"*40)
    print("NEW SHEAF CALCULATION (3-Node Chain)")
    print("#"*40)
    
    # 1. Inputs
    try:
        v_dims = [int(input(f"Dim of Vertex {i}: ")) for i in range(3)]
        e_dims = [int(input(f"Dim of Edge {i}: ")) for i in range(2)]
    except ValueError:
        print("Invalid dimension. Exiting.")
        break

    maps = {
        (0, 0): get_matrix_input(e_dims[0], v_dims[0], "V0 -> E0"),
        (0, 1): get_matrix_input(e_dims[0], v_dims[1], "V1 -> E0"),
        (1, 1): get_matrix_input(e_dims[1], v_dims[1], "V1 -> E1"),
        (1, 2): get_matrix_input(e_dims[1], v_dims[2], "V2 -> E1")
    }

    # 2. Logic
    v_off = np.insert(np.cumsum(v_dims), 0, 0)
    e_off = np.insert(np.cumsum(e_dims), 0, 0)
    delta = np.zeros((sum(e_dims), sum(v_dims)))
    
    # Edge 0 (V0 to V1)
    delta[e_off[0]:e_off[1], v_off[0]:v_off[1]] = maps[(0, 0)]
    delta[e_off[0]:e_off[1], v_off[1]:v_off[2]] = -maps[(0, 1)]
    
    # Edge 1 (V1 to V2)
    delta[e_off[1]:e_off[2], v_off[1]:v_off[2]] = maps[(1, 1)]
    delta[e_off[1]:e_off[2], v_off[2]:v_off[3]] = -maps[(1, 2)]

    L = delta.T @ delta
    spec = np.linalg.eigvalsh(L)

    # 3. Output
    print("\nLaplacian:\n", L)
    print("Spectrum:", np.round(spec, 6))
    
    save_test_results(v_dims, e_dims, maps, L, spec)
    
    cont = input("\nRun another test? (y/n): ")
    if cont.lower() != 'y':
        break