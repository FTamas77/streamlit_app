#!/usr/bin/env python3
"""
Causal Discovery Module Test Suite

This test suite focuses specifically on testing the CausalDiscovery class 
in causal_ai/discovery.py directly, without going through CausalAnalyzer.

Key aspects tested:
1. Basic causal discovery functionality 
2. Constraint handling and filtering
3. Categorical variable encoding
4. Adjacency matrix generation
5. Error handling and edge cases

Tests the core discovery algorithms in isolation to verify they work correctly
before integration testing with CausalAnalyzer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from causal_ai.discovery import CausalDiscovery

def test_causal_discovery_smoke_test():
    """
    Smoke test: Basic causal discovery without constraints (happy path).
    Tests the CausalDiscovery class directly.
    """
    print("🧪 Running causal discovery smoke test...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    # Create simple causal chain: X -> Y -> Z
    X = np.random.normal(0, 1, n_samples)
    Y = 2.0 * X + np.random.normal(0, 0.5, n_samples)
    Z = 1.5 * Y + np.random.normal(0, 0.5, n_samples)
    
    data = pd.DataFrame({
        'X': X,
        'Y': Y, 
        'Z': Z
    })
    
    print(f"📊 Created test data with shape: {data.shape}")
    
    # Initialize discovery module directly
    discovery = CausalDiscovery()
    
    # Run discovery without constraints (smoke test)
    print("🔍 Running causal discovery...")
    success = discovery.run_discovery(data)
    
    # Assertions
    assert success, "Basic causal discovery should succeed"
    assert discovery.adjacency_matrix is not None, "Adjacency matrix should be generated"
    assert discovery.adjacency_matrix.shape == (3, 3), "Adjacency matrix should be 3x3"
    
    # Check that some causal relationships were discovered
    total_edges = np.sum(np.abs(discovery.adjacency_matrix) > 0.01)
    assert total_edges > 0, "Should discover at least some causal relationships"
    
    print(f"✅ Discovered {total_edges} causal relationships")
    print(f"📈 Adjacency matrix shape: {discovery.adjacency_matrix.shape}")
    print("🎉 Causal discovery smoke test passed!")

def test_causal_discovery_with_constraints():
    """Test causal discovery with constraints applied"""
    print("🧪 Testing causal discovery with constraints...")
    
    # Set different seed for different test data
    np.random.seed(44)
    n_samples = 800
    
    # Create test data: A -> B -> C
    A = np.random.normal(0, 1, n_samples)
    B = 1.5 * A + np.random.normal(0, 0.3, n_samples)
    C = 0.8 * A + 1.2 * B + np.random.normal(0, 0.3, n_samples)
    
    data = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C
    })
    
    print(f"📊 Created constrained test data with shape: {data.shape}")
    
    # Define constraints
    constraints = {
        "forbidden": [
            "C -> A",  # Forbid reverse causation
            "B -> A"   # Forbid reverse causation  
        ]
    }
    
    # Initialize discovery module
    discovery = CausalDiscovery()
    
    # Run discovery with constraints
    print("🔍 Running causal discovery with constraints...")
    success = discovery.run_discovery(data, constraints)
    
    # Assertions
    assert success, "Constrained causal discovery should succeed"
    assert discovery.adjacency_matrix is not None, "Adjacency matrix should be generated"
    assert discovery.adjacency_matrix.shape == (3, 3), "Adjacency matrix should be 3x3"
    
    print("✅ Constrained causal discovery completed successfully!")
    
def test_causal_discovery_categorical_encoding():
    """Test that categorical variables are properly encoded"""
    print("🧪 Testing categorical variable encoding...")
    
    np.random.seed(123)
    n_samples = 500
    
    # Create mixed data with categorical variables
    categories = ['low', 'medium', 'high']
    cat_data = np.random.choice(categories, n_samples)
    num_data = np.random.normal(0, 1, n_samples)
    outcome = np.random.normal(0, 1, n_samples)
    
    data = pd.DataFrame({
        'categorical_var': cat_data,
        'numeric_var': num_data,
        'outcome': outcome  
    })
    
    print(f"📊 Created mixed data with shape: {data.shape}")
    print(f"📊 Categorical values: {data['categorical_var'].unique()}")
    
    # Initialize discovery module
    discovery = CausalDiscovery()
    
    # Run discovery - should handle categorical encoding
    print("🔍 Running discovery with categorical variables...")
    success = discovery.run_discovery(data)
    
    # Assertions
    assert success, "Discovery with categorical variables should succeed"
    assert discovery.categorical_mappings, "Should have categorical mappings"
    assert 'categorical_var' in discovery.categorical_mappings, "Should map categorical variable"
    assert discovery.encoded_data is not None, "Should store encoded data"
    
    print(f"✅ Categorical encoding successful!")
    print(f"📋 Encoded columns: {list(discovery.encoded_data.columns)}")
    print(f"🔤 Categorical mappings: {list(discovery.categorical_mappings.keys())}")

def test_causal_discovery_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing edge cases...")
    
    discovery = CausalDiscovery()
    
    # Test with too few columns
    small_data = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
    success = discovery.run_discovery(small_data)
    assert not success, "Should fail with insufficient columns"
    
    # Test with empty data
    empty_data = pd.DataFrame()
    success = discovery.run_discovery(empty_data)
    assert not success, "Should fail with empty data"
    
    print("✅ Edge cases handled correctly!")

def test_causal_discovery_adjacency_matrix_methods():
    """Test adjacency matrix access and processing methods"""
    print("🧪 Testing adjacency matrix methods...")
    
    np.random.seed(789)
    n_samples = 600
    
    # Create simple test data
    X = np.random.normal(0, 1, n_samples) 
    Y = 1.2 * X + np.random.normal(0, 0.4, n_samples)
    
    data = pd.DataFrame({'X': X, 'Y': Y})
    
    discovery = CausalDiscovery()
    success = discovery.run_discovery(data)
    
    assert success, "Discovery should succeed"
    
    # Test adjacency matrix access
    adj_matrix = discovery.get_adjacency_matrix()
    assert adj_matrix is not None, "Should return adjacency matrix"
    assert adj_matrix.shape == (2, 2), "Matrix should be 2x2"
    
    # Test causal relationships extraction
    relationships = discovery.get_causal_relationships_with_labels()
    assert isinstance(relationships, list), "Should return list of relationships"
    
    print(f"✅ Adjacency matrix methods working correctly!")
    print(f"📊 Matrix shape: {adj_matrix.shape}")
    print(f"🔗 Found {len(relationships)} relationships")

def test_causal_discovery_main():
    """Main test function that runs all discovery tests"""
    print("🧪 Testing Causal Discovery Module...")
    print("=" * 50)
    
    test_causal_discovery_smoke_test()
    print()
    
    test_causal_discovery_with_constraints()
    print()
    
    test_causal_discovery_categorical_encoding()
    print()
    
    test_causal_discovery_edge_cases()
    print()
    
    test_causal_discovery_adjacency_matrix_methods()
    print()
    
    print("=" * 50)
    print("🎉 All Causal Discovery tests passed!")

if __name__ == "__main__":
    test_causal_discovery_main()
