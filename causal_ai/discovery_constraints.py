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

def validate_constraints_feasibility(constraints: Dict, columns: List[str]) -> Dict:
    """
    Validate if constraints are feasible for causal discovery.
    Only block on fatal issues: direct conflicts, all edges forbidden, or cycles.
    """
    if not constraints:
        return {'feasible': True, 'issues': []}

    issues = []
    warnings = []
    n_variables = len(columns)
    total_possible_edges = n_variables * (n_variables - 1)  # Directed edges

    forbidden_edges = constraints.get('forbidden_edges', [])
    required_edges = constraints.get('required_edges', [])

    # Fatal: Too many forbidden edges (all edges forbidden)
    forbidden_count = len(forbidden_edges)
    if forbidden_count >= total_possible_edges:
        issues.append(f"All edges are forbidden. Causal discovery is impossible.")

    # Fatal: Direct conflicts
    forbidden_set = set((edge[0], edge[1]) for edge in forbidden_edges)
    required_set = set((edge[0], edge[1]) for edge in required_edges)
    conflicts = forbidden_set.intersection(required_set)
    if conflicts:
        for conflict in conflicts:
            issues.append(f"Edge {conflict[0]} -> {conflict[1]} is both required and forbidden")

    # Fatal: Cycles in required edges
    if len(required_edges) > 1:
        cycle_issues = _detect_potential_cycles(required_edges)
        issues.extend(cycle_issues)

    feasible = len(issues) == 0
    return {
        'feasible': feasible,
        'issues': issues,
        'warnings': warnings
    }


def _detect_potential_cycles(required_edges: List[List[str]]) -> List[str]:
    """Detect potential cycles in required edges"""
    issues = []
    edge_dict = {}
    
    # Build adjacency representation
    for source, target in required_edges:
        if source not in edge_dict:
            edge_dict[source] = []
        edge_dict[source].append(target)
    
    # Simple cycle detection for 2-cycles and 3-cycles
    for source, target in required_edges:
        # Check for 2-cycle
        if target in edge_dict and source in edge_dict[target]:
            issues.append(f"Required edges create 2-cycle: {source} <-> {target}")
        
        # Check for 3-cycle (A->B->C->A)
        if target in edge_dict:
            for next_target in edge_dict[target]:
                if next_target in edge_dict and source in edge_dict[next_target]:
                    issues.append(f"Required edges may create 3-cycle: {source} -> {target} -> {next_target} -> {source}")
    
    return issues


def _detect_isolated_variables(constraints: Dict, columns: List[str]) -> List[str]:
    """Detect variables that become completely isolated due to constraints"""
    issues = []
    
    forbidden_edges = constraints.get('forbidden_edges', [])
    required_edges = constraints.get('required_edges', [])
    
    if not forbidden_edges:
        return issues
    
    # Count connections for each variable
    var_connections = {col: {'incoming': 0, 'outgoing': 0} for col in columns}
    
    # Count forbidden connections
    for source, target in forbidden_edges:
        if source in var_connections:
            var_connections[source]['outgoing'] += 1
        if target in var_connections:
            var_connections[target]['incoming'] += 1
    
    # Check if any variable has all connections forbidden
    for var in columns:
        max_incoming = len(columns) - 1
        max_outgoing = len(columns) - 1
        
        if (var_connections[var]['incoming'] >= max_incoming and 
            var_connections[var]['outgoing'] >= max_outgoing):
            issues.append(f"Variable '{var}' has all possible connections forbidden")
    
    return issues
