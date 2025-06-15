import numpy as np
from typing import List, Dict, Optional

# Import LiNGAM utilities if available
try:
    from lingam.utils import make_prior_knowledge
    LINGAM_UTILS_AVAILABLE = True
except ImportError:
    LINGAM_UTILS_AVAILABLE = False

def create_prior_knowledge_matrix(constraints: Dict, columns: List[str]) -> Optional[np.ndarray]:
    """
    Create a proper prior knowledge matrix following the official LiNGAM tutorial.
    
    Prior knowledge matrix elements:
    - -1: No prior knowledge available
    - 0: Variable i does NOT have a directed path to variable j  
    - 1: Variable i HAS a directed path to variable j
    
    Args:
        constraints: Dictionary with constraint types:
            - 'sink_variables': List of variable names that should be sinks (no outgoing edges)
            - 'exogenous_variables': List of variable names that should be exogenous (no incoming edges)
            - 'forbidden_edges': List of [source, target] pairs that should be forbidden
            - 'required_edges': List of [source, target] pairs that should be required
        columns: List of column names in the dataset
    
    Returns:
        Prior knowledge matrix or None if LiNGAM utils not available
    """
    if not LINGAM_UTILS_AVAILABLE:
        print("WARNING: lingam.utils not available, cannot create proper prior knowledge matrix")
        return None
    
    n_variables = len(columns)
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Convert variable names to indices
    sink_indices = []
    if 'sink_variables' in constraints:
        sink_indices = [col_to_idx[var] for var in constraints['sink_variables'] if var in col_to_idx]
    
    exogenous_indices = []
    if 'exogenous_variables' in constraints:
        exogenous_indices = [col_to_idx[var] for var in constraints['exogenous_variables'] if var in col_to_idx]
    
    # Create basic prior knowledge matrix using LiNGAM's utility
    prior_knowledge = None
    
    if sink_indices:
        prior_knowledge = make_prior_knowledge(
            n_variables=n_variables,
            sink_variables=sink_indices
        )
        print(f"DEBUG: Created prior knowledge with sink variables: {[columns[i] for i in sink_indices]}")
    
    if exogenous_indices:
        if prior_knowledge is None:
            prior_knowledge = make_prior_knowledge(
                n_variables=n_variables,
                exogenous_variables=exogenous_indices
            )
        else:
            # Combine with existing prior knowledge
            exogenous_pk = make_prior_knowledge(
                n_variables=n_variables,
                exogenous_variables=exogenous_indices
            )
            # Take the more restrictive constraint (0 overrides -1)
            prior_knowledge = np.where(
                (prior_knowledge == -1) & (exogenous_pk != -1),
                exogenous_pk,
                prior_knowledge
            )
        print(f"DEBUG: Added exogenous variables: {[columns[i] for i in exogenous_indices]}")
    
    # If no sink/exogenous variables specified, start with no prior knowledge
    if prior_knowledge is None:
        prior_knowledge = make_prior_knowledge(n_variables=n_variables)
    
    # Add forbidden edges (set to 0: no directed path)
    if 'forbidden_edges' in constraints:
        for source, target in constraints['forbidden_edges']:
            if source in col_to_idx and target in col_to_idx:
                src_idx = col_to_idx[source]
                tgt_idx = col_to_idx[target]
                prior_knowledge[src_idx, tgt_idx] = 0
                print(f"DEBUG: Forbidden edge: {source} -> {target}")
    
    # Add required edges (set to 1: has directed path)
    if 'required_edges' in constraints:
        for source, target in constraints['required_edges']:
            if source in col_to_idx and target in col_to_idx:
                src_idx = col_to_idx[source]
                tgt_idx = col_to_idx[target]
                prior_knowledge[src_idx, tgt_idx] = 1
                print(f"DEBUG: Required edge: {source} -> {target}")
    
    print(f"DEBUG: Final prior knowledge matrix:\n{prior_knowledge}")
    return prior_knowledge

def convert_edge_constraints(edge_list: List[List[str]], columns) -> Optional[np.ndarray]:
    """
    Legacy function for backward compatibility.
    Converts simple edge constraints to proper prior knowledge matrix.
    """
    constraints = {
        'forbidden_edges': edge_list
    }
    return create_prior_knowledge_matrix(constraints, columns)
