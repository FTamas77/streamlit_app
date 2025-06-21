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
            - 'forbidden_edges': List of [source, target] pairs that should be forbidden
            - 'required_edges': List of [source, target] pairs that should be required
        columns: List of column names in the dataset    Returns:
        Prior knowledge matrix or None if LiNGAM utils not available
    """
    if not LINGAM_UTILS_AVAILABLE:
        print("WARNING: lingam.utils not available, cannot create proper prior knowledge matrix")
        return None
    
    n_variables = len(columns)
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Start with no prior knowledge (all -1)
    prior_knowledge = make_prior_knowledge(n_variables=n_variables)
    
    # Validate constraints for conflicts before applying
    conflicts = _validate_constraint_conflicts(constraints, columns)
    if conflicts:
        print(f"DEBUG: WARNING - Constraint conflicts detected: {conflicts}")
        print("DEBUG: This may cause DirectLiNGAM to produce cycles")
    
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


def _validate_constraint_conflicts(constraints: Dict, columns: List[str]) -> List[str]:
    """Check for conflicts in constraints that might cause cycles"""
    conflicts = []
    
    if 'forbidden_edges' not in constraints or 'required_edges' not in constraints:
        return conflicts
    
    forbidden = constraints.get('forbidden_edges', [])
    required = constraints.get('required_edges', [])
    
    # Check for direct conflicts (same edge both forbidden and required)
    for req_edge in required:
        if req_edge in forbidden:
            conflicts.append(f"Edge {req_edge[0]} -> {req_edge[1]} is both required and forbidden")
    
    # Check for potential cycle creation
    # This is a simplified check - a full check would require graph analysis
    required_edges_set = set((edge[0], edge[1]) for edge in required)
    
    # Look for potential 2-cycles in required edges
    for source, target in required:
        reverse_edge = (target, source)
        if reverse_edge in required_edges_set:
            conflicts.append(f"Required edges create 2-cycle: {source} <-> {target}")
    
    return conflicts

def convert_edge_constraints(edge_list: List[List[str]], columns) -> Optional[np.ndarray]:
    """
    Legacy function for backward compatibility.
    Converts simple edge constraints to proper prior knowledge matrix.
    """
    constraints = {
        'forbidden_edges': edge_list
    }
    return create_prior_knowledge_matrix(constraints, columns)

def validate_constraints(constraints: Dict) -> Dict:
    """
    Validate and report potential issues in LLM-generated constraints.
    
    Returns:
        Dict with validation results:
        - 'valid': bool
        - 'conflicts': List of conflict descriptions  
        - 'warnings': List of warning messages
        - 'resolved_constraints': Dict with conflicts resolved
    """
    conflicts = []
    warnings = []
    
    forbidden_edges = set()
    required_edges = set()
    
    # Extract edges
    if 'forbidden_edges' in constraints:
        forbidden_edges = {tuple(edge) for edge in constraints['forbidden_edges']}
    
    if 'required_edges' in constraints:
        required_edges = {tuple(edge) for edge in constraints['required_edges']}
    
    # Check for direct conflicts (same edge forbidden and required)
    conflicting_edges = forbidden_edges.intersection(required_edges)
    
    for edge in conflicting_edges:
        conflicts.append({
            'type': 'direct_conflict',
            'edge': list(edge),
            'description': f"Edge {edge[0]} -> {edge[1]} is both forbidden and required"
        })
    
    # Check for potential cycles in required edges
    required_graph = {}
    for source, target in required_edges:
        if source not in required_graph:
            required_graph[source] = []
        required_graph[source].append(target)
    
    # Simple cycle detection (could be more sophisticated)
    if len(required_edges) >= 3:
        warnings.append("Multiple required edges detected - check for potential cycles")
    
    # Create resolved constraints (required edges win)
    resolved_constraints = constraints.copy()
    if conflicts:
        # Remove conflicting forbidden edges
        resolved_forbidden = [edge for edge in constraints.get('forbidden_edges', []) 
                             if tuple(edge) not in conflicting_edges]
        resolved_constraints['forbidden_edges'] = resolved_forbidden
    
    return {
        'valid': len(conflicts) == 0,
        'conflicts': conflicts,
        'warnings': warnings,
        'resolved_constraints': resolved_constraints
    }
