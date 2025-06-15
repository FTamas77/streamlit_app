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
    
    def run_discovery(self, data: pd.DataFrame, constraints: Dict = None):
        """Run causal discovery with domain constraints"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                
                if constraints:
                    from causal.discovery_constraints import create_prior_knowledge_matrix
                    
                    # Use proper DirectLiNGAM prior knowledge matrix
                    prior_knowledge = create_prior_knowledge_matrix(constraints, data.columns.tolist())
                    if prior_knowledge is not None:
                        print(f"DEBUG: Using proper prior knowledge matrix")
                        self.model = DirectLiNGAM(prior_knowledge=prior_knowledge)
                    else:
                        print("DEBUG: Could not create prior knowledge matrix, running without constraints")
                        self.model = DirectLiNGAM()
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    self.model = DirectLiNGAM()
                
                self.model.fit(data)
                self.adjacency_matrix = self.model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {self.adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{self.adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(self.adjacency_matrix, np.tril(self.adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(self.model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {self.model.causal_order_}")
                    causal_order_vars = [data.columns[i] for i in self.model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                  # Enforce required edges if specified
                if constraints and 'required_edges' in constraints:
                    self._enforce_required_edges(constraints['required_edges'], data)
            
            return True
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
            return False
    
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
