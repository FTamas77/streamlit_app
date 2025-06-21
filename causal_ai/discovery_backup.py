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
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                
                # Verify the adjacency matrix represents a valid DAG
                if self._has_cycles():
                    print("DEBUG: WARNING - DirectLiNGAM produced a graph with cycles")
                    print("DEBUG: This should not happen with proper prior knowledge constraints")
                    print("DEBUG: Attempting to create DAG by removing weakest cycle edges...")
                    self._remove_cycles()
                else:
                    print("DEBUG: ✅ DirectLiNGAM produced a valid DAG")
            
            return True
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
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
                filtered_constraints['required_edges'] = filtered_required        # Keep explanation
        if 'explanation' in constraints:
            filtered_constraints['explanation'] = constraints['explanation']
        
        print(f"DEBUG: Filtered constraints: {filtered_constraints}")
        return filtered_constraints
    
    def get_adjacency_matrix(self):
        """Post-process adjacency matrix to ensure required edges exist while maintaining DAG property"""
        if self.adjacency_matrix is None:
            return
            
        columns = list(data.columns)
        for edge in required_edges:
            if len(edge) == 2:
                from_var, to_var = edge
                if from_var in columns and to_var in columns:
                    from_idx = columns.index(from_var)
                    to_idx = columns.index(to_var)
                    
                    # Check if adding this edge would create a cycle
                    if not self._would_create_cycle(from_idx, to_idx):
                        # Force this edge to exist with minimum threshold
                        if abs(self.adjacency_matrix[from_idx, to_idx]) < 0.1:
                            self.adjacency_matrix[from_idx, to_idx] = 0.1
                            print(f"DEBUG: Required edge enforced: {from_var} -> {to_var}")
                        else:
                            print(f"DEBUG: Required edge already exists: {from_var} -> {to_var}")
                    else:
                        print(f"DEBUG: Skipping required edge {from_var} -> {to_var} - would create cycle")
    
    def _ensure_dag_property(self):
        """Ensure the adjacency matrix represents a valid DAG"""
        if self.adjacency_matrix is None:
            return
        
        print("DEBUG: Validating DAG property...")
        
        # Check for cycles and remove them
        cycles_removed = self._remove_cycles()
        if cycles_removed > 0:
            print(f"DEBUG: Removed {cycles_removed} edges to eliminate cycles")
        
        # Verify no cycles remain
        if self._has_cycles():
            print("DEBUG: WARNING - Cycles still detected, attempting topological sort fix")
            self._fix_using_topological_sort()
        
        # Final validation
        if not self._has_cycles():
            print("DEBUG: ✅ DAG property validated - no cycles detected")
        else:
            print("DEBUG: ❌ WARNING - Could not eliminate all cycles")
    
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
    
    def _remove_cycles(self) -> int:
        """Remove edges to eliminate cycles, preserving as many edges as possible"""
        if self.adjacency_matrix is None:
            return 0
        
        removed_count = 0
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while self._has_cycles() and iteration < max_iterations:
            # Find the weakest edge that's part of a cycle
            cycle_edges = self._find_cycle_edges()
            
            if not cycle_edges:
                break
            
            # Remove the weakest edge
            weakest_edge = min(cycle_edges, key=lambda x: abs(self.adjacency_matrix[x[0], x[1]]))
            from_idx, to_idx = weakest_edge
            
            print(f"DEBUG: Removing cycle edge: {self.columns[from_idx]} -> {self.columns[to_idx]} (weight: {self.adjacency_matrix[from_idx, to_idx]:.3f})")
            self.adjacency_matrix[from_idx, to_idx] = 0
            removed_count += 1
            iteration += 1
        
        return removed_count
    
    def _find_cycle_edges(self) -> List[tuple]:
        """Find edges that are part of cycles"""
        if self.adjacency_matrix is None:
            return []
        
        binary_matrix = (np.abs(self.adjacency_matrix) > 0.01).astype(int)
        n = binary_matrix.shape[0]
        cycle_edges = []
        
        # Find strongly connected components using Tarjan's algorithm
        index_counter = [0]
        stack = []
        lowlinks = [0] * n
        index = [0] * n
        on_stack = [False] * n
        index_initialized = [False] * n
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            index_initialized[node] = True
            stack.append(node)
            on_stack[node] = True
            
            # Consider successors
            for successor in range(n):
                if binary_matrix[node, successor] > 0:
                    if not index_initialized[successor]:
                        strongconnect(successor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                    elif on_stack[successor]:
                        lowlinks[node] = min(lowlinks[node], index[successor])
                        # This edge is part of a cycle
                        cycle_edges.append((node, successor))
            
            # If node is a root node, pop the stack and add edges
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                
                # If component has more than one node, all internal edges are cycle edges
                if len(component) > 1:
                    for i in component:
                        for j in component:
                            if i != j and binary_matrix[i, j] > 0:
                                cycle_edges.append((i, j))
        
        for node in range(n):
            if not index_initialized[node]:
                strongconnect(node)
        
        return list(set(cycle_edges))  # Remove duplicates
    
    def _fix_using_topological_sort(self):
        """As a last resort, reorder the matrix using topological sort"""
        if self.adjacency_matrix is None:
            return
        
        try:
            # Create a directed graph
            binary_matrix = (np.abs(self.adjacency_matrix) > 0.01).astype(int)
            n = binary_matrix.shape[0]
            
            # Calculate in-degrees
            in_degree = [0] * n
            for i in range(n):
                for j in range(n):
                    if binary_matrix[i, j] > 0:
                        in_degree[j] += 1
            
            # Find nodes with no incoming edges
            queue = []
            for i in range(n):
                if in_degree[i] == 0:
                    queue.append(i)
            
            topo_order = []
            
            while queue:
                node = queue.pop(0)
                topo_order.append(node)
                
                # For each neighbor
                for neighbor in range(n):
                    if binary_matrix[node, neighbor] > 0:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
            
            # If we couldn't process all nodes, there are cycles
            if len(topo_order) != n:
                print("DEBUG: Could not create valid topological order - removing remaining edges")
                # Keep only edges that respect the partial order we found
                new_matrix = np.zeros_like(self.adjacency_matrix)
                for i, from_node in enumerate(topo_order):
                    for j, to_node in enumerate(topo_order):
                        if i < j and self.adjacency_matrix[from_node, to_node] != 0:
                            new_matrix[from_node, to_node] = self.adjacency_matrix[from_node, to_node]
                self.adjacency_matrix = new_matrix
            
        except Exception as e:
            print(f"DEBUG: Error in topological sort fix: {e}")

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
    
    def _would_create_cycle(self, from_idx: int, to_idx: int) -> bool:
        """Check if adding an edge would create a cycle in the graph"""
        # Save current state
        original_value = self.adjacency_matrix[from_idx, to_idx]
        
        # Temporarily add the edge
        self.adjacency_matrix[from_idx, to_idx] = 0.1
        
        # Check for cycles
        has_cycle = self._has_cycles()
        
        # Restore original state
        self.adjacency_matrix[from_idx, to_idx] = original_value
        
        return has_cycle
