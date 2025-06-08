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
    dot_graph = 'digraph {\n'

    # Declare nodes with valid DOT syntax
    all_nodes = set([treatment, outcome] + (confounders or []))
    for node in all_nodes:
        dot_graph += f'"{node}";\n'

    for i, from_var in enumerate(columns):
        for j, to_var in enumerate(columns):
            if abs(adjacency_matrix[j, i]) > 0.01:
                dot_graph += f'"{from_var}" -> "{to_var}";\n'

    dot_graph += '}'
    return dot_graph
