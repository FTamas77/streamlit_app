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
        # CausalDiscovery is now stateless - no data storage
        pass
    
    def run_discovery(self, data: pd.DataFrame, constraints: Dict = None):
        """Run causal discovery with constraints and return all results"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                
                # Simple encoding of categorical variables
                working_data, categorical_mappings = self._encode_categorical_variables(data)
                
                print(f"DEBUG: Working data shape: {working_data.shape}")
                print(f"DEBUG: Working columns: {list(working_data.columns)}")
                
                if working_data.shape[1] < 2:
                    print("ERROR: Need at least 2 columns for causal discovery")
                    return None
                
                # Store column information
                columns = list(working_data.columns)
                
                print(f"DEBUG: Using working data shape: {working_data.shape}")
                
                # Apply constraints if provided
                model = None
                if constraints:
                    from causal_ai.discovery_constraints import create_prior_knowledge_matrix
                    
                    # Filter constraints to work with available columns
                    filtered_constraints = self._filter_constraints(constraints, columns)
                    
                    # Use proper DirectLiNGAM prior knowledge matrix
                    prior_knowledge = create_prior_knowledge_matrix(filtered_constraints, columns)
                    if prior_knowledge is not None:
                        print(f"DEBUG: Using prior knowledge matrix")
                        model = DirectLiNGAM(prior_knowledge=prior_knowledge)
                    else:
                        print("DEBUG: Could not create prior knowledge matrix, running without constraints")
                        model = DirectLiNGAM()
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    model = DirectLiNGAM()
                
                # Fit the model
                print("DEBUG: Fitting DirectLiNGAM model...")
                model.fit(working_data)
                adjacency_matrix = model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(adjacency_matrix, np.tril(adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {model.causal_order_}")
                    causal_order_vars = [columns[i] for i in model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                  # Check for cycles (this should not happen with proper DirectLiNGAM)
                if self._has_cycles(adjacency_matrix):
                    print("DEBUG: ⚠️  WARNING - DirectLiNGAM produced a graph with cycles!")
                    print("DEBUG: This indicates an issue with the prior knowledge matrix or constraints")
                    print("DEBUG: The original DirectLiNGAM output will be preserved without modification")
                else:
                    print("DEBUG: ✅ DirectLiNGAM produced a valid DAG")
            
            # Return a dictionary with all discovery results
            return {
                'adjacency_matrix': adjacency_matrix,
                'columns': columns,
                'categorical_mappings': categorical_mappings,
                'model': model,
                'encoded_data': working_data
            }
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _filter_constraints(self, constraints: Dict, columns: List[str]) -> Dict:
        """Filter constraints to only include available columns"""
        if not constraints:
            return {}
        
        filtered_constraints = {}
        
        # Filter forbidden edges
        if 'forbidden_edges' in constraints:
            filtered_forbidden = []
            for edge in constraints['forbidden_edges']:
                if len(edge) == 2 and edge[0] in columns and edge[1] in columns:
                    filtered_forbidden.append(edge)
            if filtered_forbidden:
                filtered_constraints['forbidden_edges'] = filtered_forbidden
        
        # Filter required edges
        if 'required_edges' in constraints:
            filtered_required = []
            for edge in constraints['required_edges']:
                if len(edge) == 2 and edge[0] in columns and edge[1] in columns:
                    filtered_required.append(edge)
            if filtered_required:
                filtered_constraints['required_edges'] = filtered_required
        
        # Keep explanation
        if 'explanation' in constraints:
            filtered_constraints['explanation'] = constraints['explanation']        
        print(f"DEBUG: Filtered constraints: {filtered_constraints}")
        return filtered_constraints
    
    def _has_cycles(self, adjacency_matrix: np.ndarray) -> bool:
        """Check if the adjacency matrix has cycles"""
        if adjacency_matrix is None:
            return False
        
        # Convert to binary adjacency matrix
        binary_matrix = (np.abs(adjacency_matrix) > 0.01).astype(int)
        n = binary_matrix.shape[0]
        
        # Use DFS to detect cycles
        color = [0] * n  # 0: white, 1: gray, 2: black
        
        def has_cycle_dfs(node):
            if color[node] == 1:  # Gray node - back edge found (cycle)
                return True
            if color[node] == 2:  # Black node - already processed
                return False
                
            color[node] = 1  # Mark as gray
            
            # Visit all neighbors
            for neighbor in range(n):
                if binary_matrix[node, neighbor] > 0:
                    if has_cycle_dfs(neighbor):
                        return True
            
            color[node] = 2  # Mark as black
            return False
          # Check for cycles starting from any unvisited node
        for i in range(n):
            if color[i] == 0:
                if has_cycle_dfs(i):
                    return True
        
        return False
    
    def _encode_categorical_variables(self, data: pd.DataFrame):
        """
        Simple categorical encoding for causal discovery.
        Converts categorical variables to numeric using simple ordinal encoding.
        Returns both the encoded data and mappings.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (encoded_data, categorical_mappings)
        """
        working_data = data.copy()
        categorical_mappings = {}
        
        print("DEBUG: Simple encoding of categorical variables...")
        
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                print(f"DEBUG: Encoding categorical column: {col}")
                
                # Get unique values and sort for consistent encoding
                unique_vals = sorted(data[col].dropna().unique())
                print(f"DEBUG: Unique values in {col}: {unique_vals}")
                
                # Simple ordinal encoding: 0, 1, 2, ...
                encoding = {val: i for i, val in enumerate(unique_vals)}
                
                # Store reverse mapping for interpretation
                reverse_mapping = {i: val for val, i in encoding.items()}
                categorical_mappings[col] = {
                    'encoding': encoding,          # 'Electric' -> 1
                    'reverse': reverse_mapping,    # 1 -> 'Electric'
                    'original_values': unique_vals
                }
                  # Apply encoding
                working_data[col] = data[col].map(encoding)
                
                print(f"DEBUG: Encoded {col} with simple mapping: {encoding}")
        
        print(f"DEBUG: Working data shape: {working_data.shape}")
        print(f"DEBUG: Working data columns: {list(working_data.columns)}")
        
        return working_data, categorical_mappings
