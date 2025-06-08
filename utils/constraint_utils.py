import numpy as np
from typing import List

def convert_edge_constraints(edge_list: List[List[str]], columns) -> np.ndarray:
    """Convert edge constraints to adjacency matrix format"""
    n_vars = len(columns)
    forbidden_matrix = np.zeros((n_vars, n_vars))
    
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    for source, target in edge_list:
        if source in col_to_idx and target in col_to_idx:
            forbidden_matrix[col_to_idx[source], col_to_idx[target]] = 1
            
    return forbidden_matrix

def adjacency_to_dowhy_dot_graph(adjacency_matrix, columns, treatment: str, outcome: str, confounders: list = None):
    """Convert adjacency matrix to DOT graph format for DoWhy - properly interpret DirectLiNGAM matrix"""
    if adjacency_matrix is None or not hasattr(adjacency_matrix, 'shape'):
        print("DEBUG: adjacency_matrix is None or has no shape")
        return None
    
    print(f"DEBUG: Creating DOT graph for columns: {columns}")
    print(f"DEBUG: Treatment: {treatment}, Outcome: {outcome}, Confounders: {confounders}")
    print(f"DEBUG: Adjacency matrix:\n{adjacency_matrix}")
        
    dot_graph = 'digraph {\n'

    # Declare nodes with valid DOT syntax
    all_nodes = set([treatment, outcome] + (confounders or []))
    for node in all_nodes:
        dot_graph += f'"{node}";\n'

    # Process adjacency matrix exactly as DirectLiNGAM produced it
    if adjacency_matrix.shape[0] == len(columns) and adjacency_matrix.shape[1] == len(columns):
        edges_added = 0
        
        for i, from_var in enumerate(columns):
            for j, to_var in enumerate(columns):
                if abs(adjacency_matrix[i, j]) > 0.01:
                    print(f"DEBUG: Adding edge: {from_var} -> {to_var} (weight: {adjacency_matrix[i, j]})")
                    dot_graph += f'"{from_var}" -> "{to_var}";\n'
                    edges_added += 1
        
        print(f"DEBUG: Total edges added: {edges_added}")
    else:
        print(f"DEBUG: Matrix dimensions don't match columns. Matrix: {adjacency_matrix.shape}, Columns: {len(columns)}")

    dot_graph += '}'
    
    print(f"DEBUG: Generated DOT graph:\n{dot_graph}")
    print("=" * 50)
    
    return dot_graph
