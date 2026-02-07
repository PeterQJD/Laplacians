import networkx as nx
import matplotlib
# Force non-interactive backend for Linux terminal stability
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import sys

def transform_graph(G):
    new_G = nx.MultiGraph()
    heavy_nodes = [n for n in G.nodes() if G.degree(n) >= 2]
    leaf_nodes = [n for n in G.nodes() if G.degree(n) == 1]
    
    for leaf in leaf_nodes:
        new_G.add_node(leaf, label=G.nodes[leaf]['label'])

    edge_to_w_node = {}
    for u, v in G.edges():
        union_label = G.nodes[u]['label'] | G.nodes[v]['label']
        w_node_name = tuple(sorted((str(u), str(v)))) 
        new_G.add_node(w_node_name, label=union_label)
        edge_to_w_node[w_node_name] = w_node_name

    for heavy in heavy_nodes:
        original_label = G.nodes[heavy]['label']
        neighbors = list(G.neighbors(heavy))
        incident_w_nodes = [edge_to_w_node[tuple(sorted((str(heavy), str(nbr))))] for nbr in neighbors]
        
        if len(incident_w_nodes) == 2:
            new_G.add_edge(incident_w_nodes[0], incident_w_nodes[1], label=original_label)
        elif len(incident_w_nodes) > 2:
            for i in range(len(incident_w_nodes)):
                u_w = incident_w_nodes[i]
                v_w = incident_w_nodes[(i + 1) % len(incident_w_nodes)]
                new_G.add_edge(u_w, v_w, label=original_label)

    for leaf in leaf_nodes:
        nbr = list(G.neighbors(leaf))[0]
        edge_key = tuple(sorted((str(leaf), str(nbr))))
        w_node = edge_to_w_node[edge_key]
        intersection_label = G.nodes[leaf]['label'] & new_G.nodes[w_node]['label']
        new_G.add_edge(leaf, w_node, label=intersection_label)

    return new_G

def get_user_graph():
    G = nx.Graph()
    print("\n--- Graph Input Configuration ---")
    
    out_name = input("Enter output filename (e.g. result): ").strip()
    if not out_name.endswith(".png"):
        out_name += ".png"

    try:
        num_v = int(input("How many vertices? (Max 6): "))
        num_e = int(input("How many edges? "))
    except ValueError:
        print("Invalid input. Please use integers.")
        sys.exit(1)

    for i in range(num_v):
        v_id = input(f"ID for vertex {i+1}: ").strip()
        label_str = input(f"  Labels for {v_id} (space separated): ")
        G.add_node(v_id, label=set(label_str.split()))

    print(f"\n--- Define {num_e} Edges ---")
    for i in range(num_e):
        while True:
            e_in = input(f"Edge {i+1} (ID1 ID2): ")
            parts = e_in.split()
            if len(parts) == 2 and parts[0] in G.nodes and parts[1] in G.nodes:
                G.add_edge(parts[0], parts[1])
                break
            print(f"Error. Available nodes: {list(G.nodes())}")
    
    return G, out_name

def export_and_draw(G_old, G_new, img_filename):
    def format_lbl(s):
        return "{" + " ".join(sorted(list(s))) + "}"

    txt_filename = img_filename.replace(".png", ".txt")
    with open(txt_filename, "w") as f:
        f.write(f"MODIFIED VERTICES:\n")
        for n, d in G_new.nodes(data=True):
            f.write(f"ID {n}: {sorted(list(d['label']))}\n")

    plt.figure(figsize=(18, 9))
    
    # 1. Initial Graph (Fixing the SyntaxError here)
    plt.subplot(1, 2, 1)
    pos_old = nx.kamada_kawai_layout(G_old)
    nx.draw(G_old, pos_old, 
            labels={n: format_lbl(G_old.nodes[n]['label']) for n in G_old.nodes()}, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=3500, 
            font_size=10)
    plt.title("Initial Graph (Labels)")

    # 2. Modified Graph
    plt.subplot(1, 2, 2)
    pos_new = nx.kamada_kawai_layout(G_new)
    nx.draw(G_new, pos_new, 
            labels={n: format_lbl(G_new.nodes[n]['label']) for n in G_new.nodes()}, 
            with_labels=True, 
            node_color='salmon', 
            node_size=3500, 
            font_size=8)
    
    edge_lbls = {(u, v): format_lbl(d['label']) for u, v, k, d in G_new.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(G_new, pos_new, edge_labels=edge_lbls, font_size=7)
    
    plt.title("Modified Graph (Node & Edge Labels)")
    plt.tight_layout()
    plt.savefig(img_filename)
    plt.close()
    print(f"\nSuccess! Files saved to:\n- {os.path.abspath(img_filename)}\n- {os.path.abspath(txt_filename)}")

if __name__ == "__main__":
    result = get_user_graph()
    if result:
        user_g, filename = result
        transformed = transform_graph(user_g)
        export_and_draw(user_g, transformed, filename)