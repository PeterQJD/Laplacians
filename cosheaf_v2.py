"""
Cosheaf Spectral Analysis on Graphs

This module computes spectral data for cosheaves defined on simple graphs.
The cosheaf structure is determined by labels on vertices, with edge labels
being the union of incident vertex labels.
"""

import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Set
import networkx as nx


class LabeledGraph:
    """A simple graph with labels on vertices."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.vertex_labels = {}  # vertex -> set of labels
        self.edge_labels = {}    # (u, v) -> set of labels
    
    def add_vertex(self, vertex: int, labels: Set[str]):
        """Add a vertex with its labels."""
        self.graph.add_node(vertex)
        self.vertex_labels[vertex] = set(labels)
    
    def add_edge(self, u: int, v: int):
        """Add an edge and compute its label as union of vertex labels."""
        if u not in self.graph or v not in self.graph:
            raise ValueError("Both vertices must exist before adding edge")
        
        self.graph.add_edge(u, v)
        edge_label = self.vertex_labels[u].union(self.vertex_labels[v])
        self.edge_labels[(min(u, v), max(u, v))] = edge_label
    
    def get_vertices(self) -> List[int]:
        """Return sorted list of vertices."""
        return sorted(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Return sorted list of edges."""
        return sorted([(min(u, v), max(u, v)) for u, v in self.graph.edges()])


class Cosheaf:
    """
    Cosheaf on a labeled graph.
    
    The cosheaf assigns vector spaces to vertices and edges, with
    extension maps from vertices to incident edges.
    """
    
    def __init__(self, labeled_graph: LabeledGraph):
        self.graph = labeled_graph
        self.vertices = labeled_graph.get_vertices()
        self.edges = labeled_graph.get_edges()
        
        # Create label-to-index mappings for each vertex and edge
        self.vertex_dims = {}
        self.edge_dims = {}
        self.vertex_label_to_idx = {}
        self.edge_label_to_idx = {}
        
        self._setup_stalks()
    
    def _setup_stalks(self):
        """Set up the dimensions and indexing for stalks."""
        # For each vertex, dimension is the number of labels
        for v in self.vertices:
            labels = sorted(self.graph.vertex_labels[v])
            self.vertex_dims[v] = len(labels)
            self.vertex_label_to_idx[v] = {label: i for i, label in enumerate(labels)}
        
        # For each edge, dimension is the number of labels (union of vertex labels)
        for e in self.edges:
            labels = sorted(self.graph.edge_labels[e])
            self.edge_dims[e] = len(labels)
            self.edge_label_to_idx[e] = {label: i for i, label in enumerate(labels)}
    
    def get_extension_map(self, vertex: int, edge: Tuple[int, int]) -> np.ndarray:
        """
        Get the extension map from a vertex to an incident edge.
        
        The extension map is an injection that maps each label at the vertex
        to the corresponding coordinate in the edge stalk.
        """
        u, v = edge
        if vertex not in [u, v]:
            raise ValueError(f"Vertex {vertex} not incident to edge {edge}")
        
        vertex_labels = self.graph.vertex_labels[vertex]
        edge_labels = self.graph.edge_labels[edge]
        
        # Create the extension matrix
        rows = self.edge_dims[edge]
        cols = self.vertex_dims[vertex]
        extension = np.zeros((rows, cols))
        
        vertex_idx_map = self.vertex_label_to_idx[vertex]
        edge_idx_map = self.edge_label_to_idx[edge]
        
        # For each label at the vertex, map it to the same label at the edge
        for label in vertex_labels:
            if label in edge_labels:
                v_idx = vertex_idx_map[label]
                e_idx = edge_idx_map[label]
                extension[e_idx, v_idx] = 1.0
        
        return extension
    
    def compute_coboundary_matrix(self) -> np.ndarray:
        """
        Compute the coboundary matrix (delta_0) of the cosheaf.
        
        Maps from vertex stalks to edge stalks via extension maps.
        Returns a matrix of size (sum of edge dims) x (sum of vertex dims).
        """
        total_vertex_dim = sum(self.vertex_dims.values())
        total_edge_dim = sum(self.edge_dims.values())
        
        delta = np.zeros((total_edge_dim, total_vertex_dim))
        
        # Create index mappings for vertices and edges
        vertex_offset = {}
        current_offset = 0
        for v in self.vertices:
            vertex_offset[v] = current_offset
            current_offset += self.vertex_dims[v]
        
        edge_offset = {}
        current_offset = 0
        for e in self.edges:
            edge_offset[e] = current_offset
            current_offset += self.edge_dims[e]
        
        # Fill in the coboundary matrix
        for edge in self.edges:
            u, v = edge
            e_offset = edge_offset[edge]
            
            # Extension map from vertex u
            ext_u = self.get_extension_map(u, edge)
            u_offset = vertex_offset[u]
            delta[e_offset:e_offset + self.edge_dims[edge], 
                  u_offset:u_offset + self.vertex_dims[u]] = ext_u
            
            # Extension map from vertex v (with negative sign for orientation)
            ext_v = self.get_extension_map(v, edge)
            v_offset = vertex_offset[v]
            delta[e_offset:e_offset + self.edge_dims[edge], 
                  v_offset:v_offset + self.vertex_dims[v]] = -ext_v
        
        return delta
    
    def compute_laplacian(self) -> np.ndarray:
        """
        Compute the cosheaf Laplacian (on vertices).
        
        L_0 = delta_0^T * delta_0
        """
        delta = self.compute_coboundary_matrix()
        laplacian = delta.T @ delta
        return laplacian
    
    def compute_spectral_data(self) -> Dict:
        """
        Compute spectral data of the cosheaf.
        
        Returns:
            Dictionary containing:
            - eigenvalues: sorted eigenvalues of the Laplacian
            - eigenvectors: corresponding eigenvectors
            - coboundary: the coboundary matrix
            - laplacian: the Laplacian matrix
            - betti_0: dimension of 0-th cohomology (kernel of delta_0)
            - betti_1: dimension of 1-st cohomology (cokernel of delta_0)
        """
        delta = self.compute_coboundary_matrix()
        laplacian = delta.T @ delta
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eigh(laplacian)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute Betti numbers
        # betti_0 = dim(ker(delta_0))
        # betti_1 = dim(coker(delta_0)) = edge_dim - rank(delta_0)
        
        tol = 1e-10
        rank_delta = np.sum(eigenvalues > tol)
        betti_0 = laplacian.shape[0] - rank_delta
        betti_1 = delta.shape[0] - rank_delta
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'coboundary': delta,
            'laplacian': laplacian,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'vertex_dims': self.vertex_dims,
            'edge_dims': self.edge_dims
        }


def create_chain_graph(labels: List[Set[str]]) -> LabeledGraph:
    """
    Create a chain (path) graph with given vertex labels.
    
    Args:
        labels: List of label sets, one per vertex
    
    Returns:
        LabeledGraph representing the chain
    """
    g = LabeledGraph()
    n = len(labels)
    
    # Add vertices
    for i in range(n):
        g.add_vertex(i, labels[i])
    
    # Add edges
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    
    return g


def main():
    """Example usage with a chain graph."""
    print("=" * 60)
    print("Cosheaf Spectral Analysis Example")
    print("=" * 60)
    
    # Create a chain on 3 vertices with labels {a}, {b}, {c}
    print("\nExample 1: Chain with labels {a}, {b}, {c}")
    print("-" * 60)
    
    labels = [{'a'}, {'b'}, {'c'}]
    graph = create_chain_graph(labels)
    
    print(f"Vertices: {graph.get_vertices()}")
    print(f"Vertex labels: {[graph.vertex_labels[v] for v in graph.get_vertices()]}")
    print(f"Edges: {graph.get_edges()}")
    print(f"Edge labels: {[graph.edge_labels[e] for e in graph.get_edges()]}")
    
    # Create cosheaf and compute spectral data
    cosheaf = Cosheaf(graph)
    spectral_data = cosheaf.compute_spectral_data()
    
    print("\nCoboundary matrix (delta_0):")
    print(spectral_data['coboundary'])
    
    print("\nLaplacian matrix (L_0):")
    print(spectral_data['laplacian'])
    
    print("\nEigenvalues:")
    print(spectral_data['eigenvalues'])
    
    print(f"\nBetti numbers:")
    print(f"  beta_0 (dim H^0): {spectral_data['betti_0']}")
    print(f"  beta_1 (dim H^1): {spectral_data['betti_1']}")
    
    # Example 2: More interesting label structure
    print("\n" + "=" * 60)
    print("Example 2: Chain with overlapping labels")
    print("-" * 60)
    
    labels2 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}]
    graph2 = create_chain_graph(labels2)
    
    print(f"Vertex labels: {[graph2.vertex_labels[v] for v in graph2.get_vertices()]}")
    print(f"Edge labels: {[graph2.edge_labels[e] for e in graph2.get_edges()]}")
    
    cosheaf2 = Cosheaf(graph2)
    spectral_data2 = cosheaf2.compute_spectral_data()
    
    print("\nCoboundary matrix (delta_0):")
    print(spectral_data2['coboundary'])
    
    print("\nLaplacian matrix (L_0):")
    print(spectral_data2['laplacian'])
    
    print("\nEigenvalues:")
    print(spectral_data2['eigenvalues'])
    
    print(f"\nBetti numbers:")
    print(f"  beta_0 (dim H^0): {spectral_data2['betti_0']}")
    print(f"  beta_1 (dim H^1): {spectral_data2['betti_1']}")


if __name__ == "__main__":
    main()