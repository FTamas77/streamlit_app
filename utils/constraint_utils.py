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

def adjacency_to_graph_string(adjacency_matrix, columns) -> str:
    """Convert adjacency matrix to DoWhy graph string format"""
    if adjacency_matrix is None:
        return ""
    
    edges = []
    
    for i, source in enumerate(columns):
        for j, target in enumerate(columns):
            if abs(adjacency_matrix[i, j]) > 0.1:  # Threshold for edge
                edges.append(f'"{source}" -> "{target}"')
    
    return "; ".join(edges) if edges else ""
