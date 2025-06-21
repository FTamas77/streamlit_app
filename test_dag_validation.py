#!/usr/bin/env python3
"""
Quick test script to verify DAG validation works
"""
import numpy as np
import pandas as pd
from causal_ai.discovery import CausalDiscovery

def test_dag_validation():
    """Test our DAG validation and cycle removal"""
    print("Testing DAG validation...")
    
    # Create test data
    data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100), 
        'C': np.random.randn(100),
        'D': np.random.randn(100)
    })
    
    # Create discovery instance
    discovery = CausalDiscovery()
    
    # Create a matrix with a known cycle: A->B->C->A
    discovery.adjacency_matrix = np.array([
        [0.0, 0.5, 0.0, 0.2],  # A -> B, A -> D
        [0.0, 0.0, 0.3, 0.0],  # B -> C
        [0.4, 0.0, 0.0, 0.0],  # C -> A (creates cycle!)
        [0.0, 0.0, 0.0, 0.0]   # D (no outgoing edges)
    ])
    discovery.columns = ['A', 'B', 'C', 'D']
    
    print("Initial matrix (with cycle):")
    print(discovery.adjacency_matrix)
    print(f"Has cycles: {discovery._has_cycles()}")
    
    # Apply DAG validation
    discovery._ensure_dag_property()
    
    print("\nAfter DAG validation:")
    print(discovery.adjacency_matrix)
    print(f"Has cycles: {discovery._has_cycles()}")
    
    print("✅ DAG validation test completed")

if __name__ == "__main__":
    test_dag_validation()
