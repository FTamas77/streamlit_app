import pandas as pd
import numpy as np
import warnings
from typing import Dict, List
import streamlit as st

# Try to import optional dependencies
try:
    from lingam import DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False

class CausalDiscovery:
    """Causal discovery algorithms and graph learning"""
    
    def __init__(self):
        self.model = None
        self.adjacency_matrix = None
        self.original_columns = None  # All columns from original data
        self.numeric_columns = None   # Only numeric columns used in discovery
        self.categorical_columns = None  # Categorical columns excluded from discovery
    
    def run_discovery(self, data: pd.DataFrame, constraints: Dict = None):
        """Run causal discovery with domain constraints"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                
                # Filter to numeric columns only - DirectLiNGAM can't handle categorical data
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
                
                print(f"DEBUG: Numeric columns ({len(numeric_columns)}): {numeric_columns}")
                print(f"DEBUG: Categorical columns ({len(categorical_columns)}) excluded: {categorical_columns}")
                
                if len(numeric_columns) < 2:
                    print("ERROR: Need at least 2 numeric columns for causal discovery")
                    return False
                
                # Use only numeric data for DirectLiNGAM
                numeric_data = data[numeric_columns].copy()
                
                # Store original column mapping for later use
                self.original_columns = list(data.columns)
                self.numeric_columns = numeric_columns
                self.categorical_columns = categorical_columns
                
                print(f"DEBUG: Using numeric data shape: {numeric_data.shape}")
                
                if constraints:
                    from causal.discovery_constraints import create_prior_knowledge_matrix
                    
                    # Filter constraints to only include numeric columns
                    filtered_constraints = self._filter_constraints_for_numeric_columns(constraints, numeric_columns)
                    
                    # Use proper DirectLiNGAM prior knowledge matrix
                    prior_knowledge = create_prior_knowledge_matrix(filtered_constraints, numeric_columns)
                    if prior_knowledge is not None:
                        print(f"DEBUG: Using filtered prior knowledge matrix for numeric columns")
                        self.model = DirectLiNGAM(prior_knowledge=prior_knowledge)
                    else:
                        print("DEBUG: Could not create prior knowledge matrix, running without constraints")
                        self.model = DirectLiNGAM()
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    self.model = DirectLiNGAM()
                
                self.model.fit(numeric_data)
                self.adjacency_matrix = self.model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {self.adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{self.adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(self.adjacency_matrix, np.tril(self.adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(self.model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {self.model.causal_order_}")
                    causal_order_vars = [numeric_columns[i] for i in self.model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                  # Enforce required edges if specified
                if constraints and 'required_edges' in constraints:
                    self._enforce_required_edges(constraints['required_edges'], numeric_data)
            
            return True
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
            return False
    
    def _filter_constraints_for_numeric_columns(self, constraints: Dict, numeric_columns: List[str]) -> Dict:
        """Filter constraints to only include numeric columns"""
        if not constraints:
            return {}
        
        filtered_constraints = {}
        
        # Filter forbidden edges
        if 'forbidden_edges' in constraints:
            filtered_forbidden = []
            for edge in constraints['forbidden_edges']:
                if len(edge) == 2 and edge[0] in numeric_columns and edge[1] in numeric_columns:
                    filtered_forbidden.append(edge)
            if filtered_forbidden:
                filtered_constraints['forbidden_edges'] = filtered_forbidden
        
        # Filter required edges
        if 'required_edges' in constraints:
            filtered_required = []
            for edge in constraints['required_edges']:
                if len(edge) == 2 and edge[0] in numeric_columns and edge[1] in numeric_columns:
                    filtered_required.append(edge)
            if filtered_required:
                filtered_constraints['required_edges'] = filtered_required
        
        # Keep explanation
        if 'explanation' in constraints:
            filtered_constraints['explanation'] = constraints['explanation']
        
        print(f"DEBUG: Filtered constraints for numeric columns: {filtered_constraints}")
        return filtered_constraints
    
    def _enforce_required_edges(self, required_edges: List[List[str]], data: pd.DataFrame):
        """Post-process adjacency matrix to ensure required edges exist"""
        if self.adjacency_matrix is None:
            return
            
        columns = list(data.columns)
        for edge in required_edges:
            if len(edge) == 2:
                from_var, to_var = edge
                if from_var in columns and to_var in columns:
                    from_idx = columns.index(from_var)
                    to_idx = columns.index(to_var)
                    # Force this edge to exist with minimum threshold
                    if abs(self.adjacency_matrix[to_idx, from_idx]) < 0.1:
                        self.adjacency_matrix[to_idx, from_idx] = 0.1

    def get_adjacency_matrix(self):
        """Get the discovered adjacency matrix"""
        return self.adjacency_matrix
