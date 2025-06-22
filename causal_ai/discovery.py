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
        self.columns = None  # Column names used in discovery
        self.categorical_mappings = {}  # Store encoding mappings for interpretation
        self.encoded_data = None  # Store the encoded data for inference
    
    def run_discovery(self, data: pd.DataFrame, constraints: Dict = None):
        """Run causal discovery with constraints"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                
                # Simple encoding of categorical variables
                working_data = self._encode_categorical_variables(data)
                
                # Store the encoded data for later use in inference
                self.encoded_data = working_data.copy()
                
                print(f"DEBUG: Working data shape: {working_data.shape}")
                print(f"DEBUG: Working columns: {list(working_data.columns)}")
                
                if working_data.shape[1] < 2:
                    print("ERROR: Need at least 2 columns for causal discovery")
                    return False
                
                # Store column information
                self.columns = list(working_data.columns)
                
                print(f"DEBUG: Using working data shape: {working_data.shape}")
                
                # Apply constraints if provided
                if constraints:
                    from causal_ai.discovery_constraints import create_prior_knowledge_matrix
                    
                    # Filter constraints to work with available columns
                    filtered_constraints = self._filter_constraints(constraints, self.columns)
                    
                    # Use proper DirectLiNGAM prior knowledge matrix
                    prior_knowledge = create_prior_knowledge_matrix(filtered_constraints, self.columns)
                    if prior_knowledge is not None:
                        print(f"DEBUG: Using prior knowledge matrix")
                        self.model = DirectLiNGAM(prior_knowledge=prior_knowledge)
                    else:
                        print("DEBUG: Could not create prior knowledge matrix, running without constraints")
                        self.model = DirectLiNGAM()
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    self.model = DirectLiNGAM()
                
                # Fit the model
                print("DEBUG: Fitting DirectLiNGAM model...")
                self.model.fit(working_data)
                self.adjacency_matrix = self.model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {self.adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{self.adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(self.adjacency_matrix, np.tril(self.adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(self.model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {self.model.causal_order_}")
                    causal_order_vars = [self.columns[i] for i in self.model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                  # Check for cycles (this should not happen with proper DirectLiNGAM)
                if self._has_cycles():
                    print("DEBUG: ⚠️  WARNING - DirectLiNGAM produced a graph with cycles!")
                    print("DEBUG: This indicates an issue with the prior knowledge matrix or constraints")
                    print("DEBUG: The original DirectLiNGAM output will be preserved without modification")
                else:
                    print("DEBUG: ✅ DirectLiNGAM produced a valid DAG")
            
            return True
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
    def _has_cycles(self) -> bool:
        """Check if the current adjacency matrix has cycles"""
        if self.adjacency_matrix is None:
            return False
        
        # Convert to binary adjacency matrix
        binary_matrix = (np.abs(self.adjacency_matrix) > 0.01).astype(int)
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
      
    def get_adjacency_matrix(self):
        """Get the discovered adjacency matrix"""
        return self.adjacency_matrix
    
    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simple categorical encoding for causal discovery.
        Converts categorical variables to numeric using simple ordinal encoding.
        Stores mappings for later interpretation.
        """
        working_data = data.copy()
        self.categorical_mappings = {}  # Reset mappings
        
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
                self.categorical_mappings[col] = {
                    'encoding': encoding,          # 'Electric' -> 1
                    'reverse': reverse_mapping,    # 1 -> 'Electric'
                    'original_values': unique_vals
                }
                
                # Apply encoding
                working_data[col] = data[col].map(encoding)
                
                print(f"DEBUG: Encoded {col} with simple mapping: {encoding}")
        
        print(f"DEBUG: Working data shape: {working_data.shape}")
        print(f"DEBUG: Working data columns: {list(working_data.columns)}")
        
        return working_data
    
    def get_causal_relationships_with_labels(self, threshold=0.01):
        """
        Get causal relationships with original categorical labels for interpretation.
        
        Args:
            threshold (float): Minimum edge weight to consider as a relationship
            
        Returns:
            List[Dict]: List of relationships with human-readable labels
        """
        if self.adjacency_matrix is None or self.columns is None:
            return []
        
        relationships = []
        
        for i, target in enumerate(self.columns):
            for j, source in enumerate(self.columns):
                if i != j and abs(self.adjacency_matrix[i, j]) > threshold:
                    weight = self.adjacency_matrix[i, j]
                    
                    # Create relationship info
                    relationship = {
                        'source': source,
                        'target': target,
                        'weight': weight,
                        'direction': 'positive' if weight > 0 else 'negative',
                        'strength': 'strong' if abs(weight) > 0.3 else 'moderate' if abs(weight) > 0.1 else 'weak'
                    }
                    
                    # Add categorical interpretation if available
                    if source in self.categorical_mappings:
                        relationship['source_categories'] = self.categorical_mappings[source]['original_values']
                        relationship['source_type'] = 'categorical'
                    else:
                        relationship['source_type'] = 'numeric'
                        
                    if target in self.categorical_mappings:
                        relationship['target_categories'] = self.categorical_mappings[target]['original_values']
                        relationship['target_type'] = 'categorical'
                    else:
                        relationship['target_type'] = 'numeric'
                    
                    relationships.append(relationship)
        
        return relationships
    
    def decode_categorical_value(self, column, encoded_value):
        """
        Convert an encoded categorical value back to its original label.
        
        Args:
            column (str): Column name
            encoded_value (int): Encoded numeric value
            
        Returns:
            str: Original categorical label, or the encoded_value if not categorical
        """
        if column in self.categorical_mappings:
            reverse_mapping = self.categorical_mappings[column]['reverse']
            return reverse_mapping.get(encoded_value, f"Unknown({encoded_value})")
        return encoded_value
    
    def get_categorical_info(self):
        """
        Get information about all categorical variables and their encodings.
        
        Returns:
            Dict: Information about categorical encodings
        """
        return self.categorical_mappings
