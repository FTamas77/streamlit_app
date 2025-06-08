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
    """Convert adjacency matrix to DOT graph format for DoWhy - only call when matrix exists"""
    if adjacency_matrix is None or not hasattr(adjacency_matrix, 'shape'):
        return None
        
    dot_graph = 'digraph {\n'

    # Declare nodes with valid DOT syntax
    all_nodes = set([treatment, outcome] + (confounders or []))
    for node in all_nodes:
        dot_graph += f'"{node}";\n'

    # Process adjacency matrix
    if adjacency_matrix.shape[0] == len(columns) and adjacency_matrix.shape[1] == len(columns):
        for i, from_var in enumerate(columns):
            for j, to_var in enumerate(columns):
                if abs(adjacency_matrix[i, j]) > 0.01:
                    dot_graph += f'"{from_var}" -> "{to_var}";\n'

    dot_graph += '}'
    return dot_graph
