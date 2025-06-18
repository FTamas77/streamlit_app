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
        self.numeric_columns = None   # Only numeric columns used in discovery        self.categorical_columns = None  # Categorical columns excluded from discovery
    
    def run_discovery(self, data: pd.DataFrame, constraints: Dict = None):
        """Run causal discovery with domain constraints"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {data.shape}")
                print(f"DEBUG: Data columns: {list(data.columns)}")
                
                # Create encoded dataset for causal discovery
                encoded_data, column_mapping = self._encode_data_for_causal_discovery(data)
                
                print(f"DEBUG: Encoded data shape: {encoded_data.shape}")
                print(f"DEBUG: Encoded columns: {list(encoded_data.columns)}")
                print(f"DEBUG: Column mapping: {column_mapping}")
                
                if encoded_data.shape[1] < 2:
                    print("ERROR: Need at least 2 columns for causal discovery after encoding")
                    return False                
                # Store column information for later use
                self.original_columns = list(data.columns)
                self.encoded_columns = list(encoded_data.columns)
                self.column_mapping = column_mapping
                
                # Separate numeric and categorical from original data for display purposes
                original_numeric = data.select_dtypes(include=[np.number]).columns.tolist()
                original_categorical = data.select_dtypes(exclude=[np.number]).columns.tolist()
                self.numeric_columns = original_numeric
                self.categorical_columns = original_categorical
                
                print(f"DEBUG: Using encoded data shape: {encoded_data.shape}")
                
                if constraints:
                    from causal.discovery_constraints import create_prior_knowledge_matrix
                    
                    # Filter constraints to work with encoded columns
                    filtered_constraints = self._filter_constraints_for_encoded_columns(constraints, self.encoded_columns)
                    
                    # Use proper DirectLiNGAM prior knowledge matrix
                    prior_knowledge = create_prior_knowledge_matrix(filtered_constraints, self.encoded_columns)
                    if prior_knowledge is not None:
                        print(f"DEBUG: Using filtered prior knowledge matrix for numeric columns")
                        self.model = DirectLiNGAM(prior_knowledge=prior_knowledge)
                    else:
                        print("DEBUG: Could not create prior knowledge matrix, running without constraints")
                        self.model = DirectLiNGAM()
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    self.model = DirectLiNGAM()                
                self.model.fit(encoded_data)
                self.adjacency_matrix = self.model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {self.adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{self.adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(self.adjacency_matrix, np.tril(self.adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(self.model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {self.model.causal_order_}")
                    causal_order_vars = [self.encoded_columns[i] for i in self.model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")                  # Enforce required edges if specified
                if constraints and 'required_edges' in constraints:
                    self._enforce_required_edges(constraints['required_edges'], encoded_data)
            
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
    
    def _encode_data_for_causal_discovery(self, data: pd.DataFrame):
        """
        Encode categorical variables as numeric for causal discovery.
        This allows DirectLiNGAM to work with categorical causes.
        
        Returns:
        - encoded_data: DataFrame with categorical variables encoded as numeric
        - column_mapping: Dict mapping encoded column names to original values
        """
        encoded_data = data.copy()
        column_mapping = {}
        
        print("DEBUG: Encoding categorical variables for causal discovery...")
        
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                print(f"DEBUG: Encoding categorical column: {col}")
                
                # Get unique values
                unique_vals = data[col].dropna().unique()
                print(f"DEBUG: Unique values in {col}: {unique_vals}")
                
                # Create smart encoding based on column content
                if col.lower() in ['fuel_type', 'fuel', 'energy_type']:
                    # For fuel types, prioritize by environmental impact
                    encoding = self._encode_fuel_type(unique_vals)
                elif col.lower() in ['vehicle_type', 'vehicle', 'transport_type']:
                    # For vehicle types, encode by typical capacity/emissions
                    encoding = self._encode_vehicle_type(unique_vals)
                elif col.lower() in ['region', 'location', 'area']:
                    # For regions, use alphabetical order
                    encoding = {val: i for i, val in enumerate(sorted(unique_vals))}
                else:
                    # Default: alphabetical order
                    encoding = {val: i for i, val in enumerate(sorted(unique_vals))}
                
                # Apply encoding
                encoded_col_name = f"{col}_Code"
                encoded_data[encoded_col_name] = data[col].map(encoding)
                
                # Drop original categorical column
                encoded_data = encoded_data.drop(columns=[col])
                
                # Store mapping for later reference
                column_mapping[encoded_col_name] = {
                    'original_column': col,
                    'encoding': encoding,
                    'reverse_encoding': {v: k for k, v in encoding.items()}
                }
                
                print(f"DEBUG: Encoded {col} as {encoded_col_name} with mapping: {encoding}")
        
        print(f"DEBUG: Final encoded data shape: {encoded_data.shape}")
        print(f"DEBUG: Final encoded columns: {list(encoded_data.columns)}")
        
        return encoded_data, column_mapping
    
    def _encode_fuel_type(self, unique_vals):
        """Smart encoding for fuel types based on environmental impact"""
        # Order by environmental impact (lower values = cleaner)
        fuel_priority = {
            'Electric': 0,
            'electric': 0,
            'Electric_Van': 0,
            'Hybrid': 1,
            'hybrid': 1,
            'Gasoline': 2,
            'gasoline': 2,
            'Petrol': 2,
            'petrol': 2,
            'Diesel': 3,
            'diesel': 3
        }
        
        encoding = {}
        next_code = max(fuel_priority.values()) + 1 if fuel_priority.values() else 0
        
        for val in unique_vals:
            if val in fuel_priority:
                encoding[val] = fuel_priority[val]
            else:
                encoding[val] = next_code
                next_code += 1
        
        return encoding
    
    def _encode_vehicle_type(self, unique_vals):
        """Smart encoding for vehicle types based on typical capacity/emissions"""
        # Order by typical capacity/emissions (lower values = smaller/cleaner)
        vehicle_priority = {
            'Electric_Van': 0,
            'electric_van': 0,
            'Van': 1,
            'van': 1,
            'Car': 1,
            'car': 1,
            'Truck': 2,
            'truck': 2,
            'Semi': 3,
            'semi': 3
        }
        
        encoding = {}
        next_code = max(vehicle_priority.values()) + 1 if vehicle_priority.values() else 0
        
        for val in unique_vals:
            if val in vehicle_priority:
                encoding[val] = vehicle_priority[val]
            else:
                encoding[val] = next_code
                next_code += 1
        
        return encoding
    
    def _filter_constraints_for_encoded_columns(self, constraints, encoded_columns):
        """Filter constraints to work with encoded column names"""
        if not constraints:
            return constraints
        
        filtered = {}
        
        # Filter forbidden edges
        if 'forbidden_edges' in constraints:
            filtered['forbidden_edges'] = []
            for edge in constraints['forbidden_edges']:
                # Try to map original column names to encoded names
                mapped_edge = self._map_edge_to_encoded(edge, encoded_columns)
                if mapped_edge:
                    filtered['forbidden_edges'].append(mapped_edge)
        
        # Filter required edges  
        if 'required_edges' in constraints:
            filtered['required_edges'] = []
            for edge in constraints['required_edges']:
                mapped_edge = self._map_edge_to_encoded(edge, encoded_columns)
                if mapped_edge:
                    filtered['required_edges'].append(mapped_edge)
        
        return filtered
    
    def _map_edge_to_encoded(self, edge, encoded_columns):
        """Map an edge from original column names to encoded column names"""
        source, target = edge
        
        # Check if columns exist as-is (for numeric columns)
        if source in encoded_columns and target in encoded_columns:
            return [source, target]
        
        # Check for encoded versions
        source_encoded = f"{source}_Code" if f"{source}_Code" in encoded_columns else source
        target_encoded = f"{target}_Code" if f"{target}_Code" in encoded_columns else target
        
        if source_encoded in encoded_columns and target_encoded in encoded_columns:
            return [source_encoded, target_encoded]
        
        # Edge references columns not in the encoded dataset
        print(f"DEBUG: Skipping constraint edge {source} -> {target} (columns not found in encoded data)")
        return None
