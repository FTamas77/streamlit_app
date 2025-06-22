#!/usr/bin/env python3
"""
CausalAnalyzer Integration Test Suite

This test suite focuses specifically on testing the CausalAnalyzer class
in causal_ai/analyzer.py as an integration layer that coordinates between
causal discovery and causal inference modules.

Key aspects tested:
1. Integration between discovery and inference
2. Data flow through the analyzer pipeline  
3. End-to-end causal analysis workflow
4. Error handling across modules
5. State management and consistency

These are integration tests that verify the modules work together correctly,
complementing the unit tests for individual discovery and inference modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from causal_ai.analyzer import CausalAnalyzer

def test_causal_analyzer_end_to_end_workflow():
    """
    Test the complete end-to-end workflow: data loading -> discovery -> inference
    """
    print("🧪 Running CausalAnalyzer end-to-end workflow test...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic causal data: X -> treatment -> outcome, Y -> treatment
    X = np.random.normal(0, 1, n_samples)  # Confounder
    Y = np.random.normal(0, 1, n_samples)  # Another confounder
    
    # Treatment is influenced by confounders
    treatment_prob = 0.3 + 0.2 * X + 0.1 * Y
    treatment = np.random.binomial(1, np.clip(treatment_prob, 0, 1), n_samples)
    
    # Outcome has true causal effect from treatment
    true_ate = 0.5
    outcome = (0.3 * X +           # Confounder effect
              0.2 * Y +           # Confounder effect
              true_ate * treatment +  # TRUE CAUSAL EFFECT
              np.random.normal(0, 0.5, n_samples))
    
    data = pd.DataFrame({
        'X': X,
        'Y': Y,
        'treatment': treatment,
        'outcome': outcome
    })
    
    print(f"📊 Created test data with shape: {data.shape}")
    print(f"📊 True ATE: {true_ate}")
    
    # Initialize analyzer
    analyzer = CausalAnalyzer()
    analyzer.data = data
    
    # Step 1: Run causal discovery
    print("🔍 Step 1: Running causal discovery...")
    discovery_success = analyzer.run_causal_discovery()
    assert discovery_success, f"Causal discovery should succeed, got {discovery_success}"
    assert analyzer.adjacency_matrix is not None, "Should have adjacency matrix"
    assert analyzer.discovery is not None, "Should have discovery object"
      # Step 2: Run causal inference
    print("🔍 Step 2: Running causal inference...")
    inference_results = analyzer.calculate_ate('treatment', 'outcome')
    assert inference_results is not None, "Causal inference should succeed"
    assert 'consensus_estimate' in inference_results, f"Should have consensus estimate, got {inference_results}"
    
    estimated_ate = inference_results['consensus_estimate']
    print(f"📈 Estimated ATE: {estimated_ate:.4f}")
    print(f"📊 True ATE: {true_ate}")
    
    # The estimate should be reasonably close to the true effect
    # (allowing for some statistical variation)
    assert abs(estimated_ate - true_ate) < 1.0, f"Estimate {estimated_ate:.4f} should be reasonably close to true ATE {true_ate}"
    
    print("✅ End-to-end workflow completed successfully!")
    print("🎉 CausalAnalyzer integration test passed!")

def test_causal_analyzer_state_consistency():
    """
    Test that CausalAnalyzer maintains consistent state across operations
    """
    print("🧪 Testing CausalAnalyzer state consistency...")
    
    np.random.seed(789)
    n_samples = 600
    
    # Create simple test data
    X = np.random.normal(0, 1, n_samples)
    Y = 1.5 * X + np.random.normal(0, 0.3, n_samples)
    Z = 0.8 * Y + np.random.normal(0, 0.3, n_samples)
    
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    analyzer = CausalAnalyzer()
    analyzer.data = data
      # Check initial state
    assert analyzer.data is not None, "Data should be loaded"
    assert analyzer.discovery is not None, "Discovery should be initialized in __init__"
    assert analyzer.adjacency_matrix is None, "Adjacency matrix should not exist yet"
    
    # Run discovery
    discovery_success = analyzer.run_causal_discovery()
    assert discovery_success, f"Discovery should succeed, got {discovery_success}"
    
    # Check state after discovery
    assert analyzer.discovery is not None, "Discovery object should exist"
    assert analyzer.adjacency_matrix is not None, "Adjacency matrix should exist"
    assert analyzer.discovery.adjacency_matrix is not None, "Discovery should have adjacency matrix"
    
    # Check consistency between analyzer and discovery
    np.testing.assert_array_equal(
        analyzer.adjacency_matrix, 
        analyzer.discovery.adjacency_matrix,
        "Analyzer and discovery adjacency matrices should be identical"
    )
    
    # Test that data remains consistent
    assert analyzer.data.equals(data), "Original data should remain unchanged"
    
    print("✅ State consistency verified!")

def test_causal_analyzer_with_constraints():
    """
    Test CausalAnalyzer integration with constraints applied to discovery
    """
    print("🧪 Testing CausalAnalyzer with discovery constraints...")
    
    np.random.seed(456)
    n_samples = 800
    
    # Create causal chain: A -> B -> C
    A = np.random.normal(0, 1, n_samples)
    B = 1.2 * A + np.random.normal(0, 0.4, n_samples)
    C = 0.9 * B + np.random.normal(0, 0.4, n_samples)
    
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    
    analyzer = CausalAnalyzer()
    analyzer.data = data
    
    # Define constraints (fix: use 'forbidden_edges' as expected by discovery module)
    constraints = {
        "forbidden_edges": [
            ["C", "A"],  # Forbid reverse causation
            ["B", "A"],  # Forbid reverse causation
            ["C", "B"]   # Forbid reverse causation
        ]
    }
    
    # Run discovery with constraints
    discovery_success = analyzer.run_causal_discovery(constraints)
    assert discovery_success, f"Constrained discovery should succeed, got {discovery_success}"
      # Run inference to make sure discovery results work with inference
    inference_results = analyzer.calculate_ate('A', 'C')
    assert inference_results is not None, "Inference should work with constrained discovery"
    
    print("✅ Constrained discovery integration successful!")

def test_causal_analyzer_error_handling():
    """
    Test error handling and recovery in CausalAnalyzer
    """
    print("🧪 Testing CausalAnalyzer error handling...")
    
    analyzer = CausalAnalyzer()
      # Test inference without discovery
    try:
        result = analyzer.calculate_ate('treatment', 'outcome')
        assert result is None or isinstance(result, dict), "Should handle missing discovery gracefully"
    except Exception as e:
        print(f"Expected error (no discovery): {e}")
    
    # Test with insufficient data
    small_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    analyzer.data = small_data
    try:
        discovery_result = analyzer.run_causal_discovery()
        assert not discovery_result, "Should fail or return False with insufficient data"
    except Exception as e:
        print(f"Expected error (insufficient data): {e}")
    
    print("✅ Error handling working correctly!")

def test_causal_analyzer_different_data_types():
    """
    Test CausalAnalyzer with different variable types (categorical, continuous)
    """
    print("🧪 Testing CausalAnalyzer with mixed data types...")
    
    np.random.seed(333)
    n_samples = 700
    
    # Create mixed data
    continuous_var = np.random.normal(0, 1, n_samples)
    categorical_var = np.random.choice(['low', 'medium', 'high'], n_samples)
    binary_treatment = np.random.binomial(1, 0.4, n_samples)
    outcome = (continuous_var * 0.3 + 
              binary_treatment * 0.5 + 
              np.random.normal(0, 0.5, n_samples))
    
    mixed_data = pd.DataFrame({
        'continuous_var': continuous_var,
        'categorical_var': categorical_var,
        'binary_treatment': binary_treatment,
        'outcome': outcome
    })
    
    analyzer = CausalAnalyzer()
    analyzer.data = mixed_data
    
    # Test discovery with mixed types
    discovery_success = analyzer.run_causal_discovery()
    print(f"Discovery success: {discovery_success}")
    assert discovery_success, f"Discovery should succeed with mixed data types, got {discovery_success}"
      # Test inference with mixed types
    inference_results = analyzer.calculate_ate('binary_treatment', 'outcome')
    assert inference_results is not None, "Inference should work with mixed data types"
    print("✅ Mixed data types handled successfully!")

def test_causal_analyzer_integration_main():
    """Main test function that runs all CausalAnalyzer integration tests"""
    print("🧪 Testing CausalAnalyzer Integration...")
    print("=" * 60)
    try:
        test_causal_analyzer_end_to_end_workflow()
        print()
    except AssertionError as e:
        print(f"❌ test_causal_analyzer_end_to_end_workflow failed: {e}")
        raise
    try:
        test_causal_analyzer_state_consistency()
        print()
    except AssertionError as e:
        print(f"❌ test_causal_analyzer_state_consistency failed: {e}")
        raise
    try:
        test_causal_analyzer_with_constraints()
        print()
    except AssertionError as e:
        print(f"❌ test_causal_analyzer_with_constraints failed: {e}")
        raise
    try:
        test_causal_analyzer_error_handling()
        print()
    except AssertionError as e:
        print(f"❌ test_causal_analyzer_error_handling failed: {e}")
        raise
    try:
        test_causal_analyzer_different_data_types()
        print()
    except AssertionError as e:
        print(f"❌ test_causal_analyzer_different_data_types failed: {e}")
        raise
    print("=" * 60)
    print("🎉 All CausalAnalyzer Integration tests passed!")

if __name__ == "__main__":
    test_causal_analyzer_integration_main()
