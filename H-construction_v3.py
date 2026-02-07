import networkx as nx
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import os
import sys

# --- 1. DISCRIMINATION ANALYSIS (FOR GRAPH G) ---

def analyze_initial_weighted_graph(G):
    """Computes the spectrum based on intersection sizes between vertices."""
    nodes = list(G.nodes())
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    total_intersection_weight = 0
    
    for u, v in G.edges():
        u_idx, v_idx = nodes.index(u), nodes.index(v)
        # Weight is the cardinality of the intersection
        weight = len(G.nodes[u]['label'] & G.nodes[v]['label'])
        adj_matrix[u_idx, v_idx] = weight
        adj_matrix[v_idx, u_idx] = weight
        total_intersection_weight += weight
        
    weighted_spectrum = np.linalg.eigvalsh(adj_matrix)
    return weighted_spectrum, total_intersection_weight

# --- 2. H-CONSTRUCTION LOGIC ---

def build_restriction_map(v_set, e_set):
    """Builds identity-based restriction map."""
    v_list = sorted(list(v_set))
    e_list = sorted(list(e_set))
    if not e_list: return np.zeros((0, len(v_list)))
    matrix = np.zeros((len(e_list), len(v_list)))
    for i, e_elem in enumerate(e_list):
        if e_elem in v_list:
            matrix[i, v_list.index(e_elem)] = 1.0
    return matrix

def transform_graph(G):
    """The H-construction transformation."""
    H = nx.MultiGraph()
    heavy_nodes = [n for n in G.nodes() if G.degree(n) >= 2]
    leaf_nodes = [n for n in G.nodes() if G.degree(n) == 1]
    
    for leaf in leaf_nodes:
        H.add_node(leaf, label=G.nodes[leaf]['label'])

    edge_to_w_node = {}
    for u, v in G.edges():
        union_label = G.nodes[u]['label'] | G.nodes[v]['label']
        w_node_name = tuple(sorted((str(u), str(v)))) 
        H.add_node(w_node_name, label=union_label)
        edge_to_w_node[w_node_name] = w_node_name

    for heavy in heavy_nodes:
        original_label = G.nodes[heavy]['label']
        neighbors = list(G.neighbors(heavy))
        incident_w_nodes = [edge_to_w_node[tuple(sorted((str(heavy), str(nbr))))] for nbr in neighbors]
        
        if len(incident_w_nodes) == 2:
            H.add_edge(incident_w_nodes[0], incident_w_nodes[1], label=original_label)
        elif len(incident_w_nodes) > 2:
            for i in range(len(incident_w_nodes)):
                u_w = incident_w_nodes[i]
                v_w = incident_w_nodes[(i + 1) % len(incident_w_nodes)]
                H.add_edge(u_w, v_w, label=original_label)

    for leaf in leaf_nodes:
        nbr = list(G.neighbors(leaf))[0]
        w_node = edge_to_w_node[tuple(sorted((str(leaf), str(nbr))))]
        H.add_edge(leaf, w_node, label=G.nodes[leaf]['label'] & H.nodes[w_node]['label'])

    return H

# --- 3. SHEAF LAPLACIAN CALCULATIONS ---

def compute_sheaf_data(H):
    nodes = list(H.nodes())
    edges = list(H.edges(keys=True, data=True))
    v_dims = [len(H.nodes[n]['label']) for n in nodes]
    e_dims = [len(d['label']) for u, v, k, d in edges]
    
    total_v_dim = sum(v_dims)
    if total_v_dim == 0: return None

    delta = np.zeros((sum(e_dims), total_v_dim))
    v_off = np.insert(np.cumsum(v_dims), 0, 0)
    e_off = np.insert(np.cumsum(e_dims), 0, 0)

    for i, (u, v, k, d) in enumerate(edges):
        if e_dims[i] > 0:
            u_idx, v_idx = nodes.index(u), nodes.index(v)
            delta[e_off[i]:e_off[i+1], v_off[u_idx]:v_off[u_idx+1]] = build_restriction_map(H.nodes[u]['label'], d['label'])
            delta[e_off[i]:e_off[i+1], v_off[v_idx]:v_off[v_idx+1]] = -1.0 * build_restriction_map(H.nodes[v]['label'], d['label'])

    L = delta.T @ delta
    spec = np.linalg.eigvalsh(L)
    null_space = scipy.linalg.null_space(L)
    return {"L": L, "spectrum": spec, "energy": np.sum(spec**2), "h0": null_space.shape[1], "basis": null_space}

# --- 4. INPUT AND OUTPUT ---

def get_user_graph():
    G = nx.Graph()
    print("\n" + "="*40 + "\n   H-CONSTRUCTION & DISCRIMINATION\n" + "="*40)
    out_name = input("Enter output filename: ").strip()
    if not out_name.endswith(".png"): out_name += ".png"

    try:
        num_v = int(input("How many vertices? (Max 6): "))
        num_e = int(input("How many edges? "))
    except ValueError: sys.exit("Error: Please enter integers.")

    print("\nNote: Enter labels separated by SPACES (e.g., 'a b c')")
    for i in range(num_v):
        v_id = input(f"ID for vertex {i+1}: ").strip()
        label_str = input(f"  Labels for {v_id}: ")
        G.add_node(v_id, label=set(label_str.split()))

    for i in range(num_e):
        while True:
            e_in = input(f"Edge {i+1} (ID1 ID2): ")
            parts = e_in.split()
            if len(parts) == 2 and parts[0] in G.nodes and parts[1] in G.nodes:
                G.add_edge(parts[0], parts[1])
                break
            print("Invalid nodes. Try again.")
    return G, out_name

def export_results(G_old, H, filename, spectral_h, weighted_spec, total_weight):
    def format_lbl(s): return "{" + " ".join(sorted(list(s))) + "}"
    txt_filename = filename.replace(".png", ".txt")
    
    with open(txt_filename, "w") as f:
        f.write("--- DISCRIMINATION ANALYSIS (INITIAL GRAPH G) ---\n")
        f.write(f"WEIGHTED ADJACENCY SPECTRUM: {np.round(weighted_spec, 4)}\n")
        f.write(f"TOTAL INTERSECTION WEIGHT: {total_weight}\n")
        f.write("(Weights defined by |Vi ∩ Vj|)\n\n")
        
        f.write("--- H-CONSTRUCTION RESULTS ---\n\n")
        f.write("INITIAL GRAPH G VERTICES:\n")
        for n, d in G_old.nodes(data=True):
            f.write(f"{n}: {format_lbl(d['label'])}\n")
        
        f.write("\nCONSTRUCTED GRAPH H VERTICES:\n")
        for n, d in H.nodes(data=True):
            f.write(f"{n}: {format_lbl(d['label'])}\n")
            
        if spectral_h:
            f.write(f"\n--- SPECTRAL DATA FOR H ---\n")
            f.write(f"ENERGY: {round(spectral_h['energy'], 6)}\n")
            f.write(f"H0 DIMENSION: {spectral_h['h0']}\n")
            f.write(f"SPECTRUM: {np.round(spectral_h['spectrum'], 4)}\n")
            f.write(f"\nLAPLACIAN MATRIX:\n{spectral_h['L']}\n")
            if spectral_h['h0'] > 0:
                f.write(f"\nBASIS FOR GLOBAL SECTIONS (Null Space):\n")
                f.write(f"{np.round(spectral_h['basis'], 4)}\n")

    # Plot
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    pos_old = nx.kamada_kawai_layout(G_old)
    nx.draw(G_old, pos_old, labels={n: format_lbl(G_old.nodes[n]['label']) for n in G_old.nodes()}, 
            with_labels=True, node_color='lightblue', node_size=3500, font_size=9)
    plt.title("Initial Graph G")

    plt.subplot(1, 2, 2)
    pos_h = nx.kamada_kawai_layout(H)
    nx.draw(H, pos_h, labels={n: format_lbl(H.nodes[n]['label']) for n in H.nodes()}, 
            with_labels=True, node_color='salmon', node_size=3500, font_size=7)
    edge_lbls = {(u, v): format_lbl(d['label']) for u, v, k, d in H.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(H, pos_h, edge_labels=edge_lbls, font_size=7)
    plt.title("Transformed Graph H")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    user_g, filename = get_user_graph()
    
    # 1. Analyze initial G
    w_spec, total_w = analyze_initial_weighted_graph(user_g)
    
    # 2. Build H and analyze
    H_graph = transform_graph(user_g)
    spectral_data = compute_sheaf_data(H_graph)
    
    # 3. Export
    export_results(user_g, H_graph, filename, spectral_data, w_spec, total_w)
    print(f"\nSuccess! Results saved to {filename} and {filename.replace('.png', '.txt')}")