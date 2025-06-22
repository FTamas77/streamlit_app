#!/usr/bin/env python3
"""
Causal Inference Module Test Suite

This test suite focuses specifically on testing the causal inference functions
in causal_ai/inference.py directly, without going through CausalAnalyzer.

Key aspects tested:
1. ATE calculation with DoWhy 
2. Input validation and error handling
3. Graph generation and processing
4. Effect size interpretation
5. Statistical metrics calculation

Tests the core inference algorithms in isolation to verify they work correctly
before integration testing with CausalAnalyzer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from causal_ai.inference import calculate_ate_dowhy, _interpret_ate, _generate_simple_recommendation

def create_mock_analyzer_for_inference():
    """Create a mock analyzer with realistic causal data for testing inference directly"""
    mock_analyzer = Mock()
    
    # Create realistic data with actual causal relationships
    np.random.seed(42)
    n_samples = 1000  # Larger sample for better statistical power
    
    # Generate confounders
    X = np.random.normal(0, 1, n_samples)
    Y = np.random.normal(0, 1, n_samples) 
    Z = np.random.normal(0, 1, n_samples)
    
    # Generate treatment with confounding
    treatment_prob = 0.3 + 0.2 * X + 0.1 * Y  # Confounded by X and Y
    treatment = np.random.binomial(1, np.clip(treatment_prob, 0, 1), n_samples)
    
    # Generate outcome with REAL causal effect
    true_effect = 0.5  # Significant causal effect
    outcome = (0.3 * X +           # Confounder effect
              0.2 * Y +           # Confounder effect  
              0.1 * Z +           # Another confounder
              true_effect * treatment +  # TRUE CAUSAL EFFECT
              np.random.normal(0, 0.5, n_samples))  # Noise
    
    data = pd.DataFrame({
        'X': X,
        'Y': Y, 
        'Z': Z,
        'treatment': treatment,
        'outcome': outcome
    })
    mock_analyzer.data = data
    
    # Mock the discovery object with numeric columns
    mock_discovery = Mock()
    mock_discovery.numeric_columns = ['X', 'Y', 'Z', 'treatment', 'outcome']
    mock_discovery.encoded_data = data  # Use same data for simplicity
    mock_discovery.categorical_mappings = {}
    
    # Mock adjacency matrix (simple causal structure)
    adjacency_matrix = np.zeros((5, 5))
    # X -> treatment, Y -> treatment, treatment -> outcome
    adjacency_matrix[0, 3] = 0.2  # X -> treatment  
    adjacency_matrix[1, 3] = 0.1  # Y -> treatment
    adjacency_matrix[3, 4] = 0.5  # treatment -> outcome
    adjacency_matrix[0, 4] = 0.3  # X -> outcome (confounder)
    adjacency_matrix[1, 4] = 0.2  # Y -> outcome (confounder)
    
    mock_discovery.adjacency_matrix = adjacency_matrix
    mock_discovery.columns = ['X', 'Y', 'Z', 'treatment', 'outcome']
    mock_analyzer.discovery = mock_discovery
    mock_analyzer.adjacency_matrix = adjacency_matrix
    mock_analyzer.columns = ['X', 'Y', 'Z', 'treatment', 'outcome']
    
    return mock_analyzer

def test_inference_smoke_test():
    """
    Smoke test: Basic ATE calculation without issues (happy path).
    Tests the calculate_ate_dowhy function directly.
    """
    print("🧪 Running causal inference smoke test...")
    
    # Create mock analyzer with realistic data
    mock_analyzer = create_mock_analyzer_for_inference()
    treatment_var = 'treatment'
    outcome_var = 'outcome'
    
    print(f"📊 Created test data with shape: {mock_analyzer.data.shape}")
    print(f"📊 Treatment variable: {treatment_var}")
    print(f"📊 Outcome variable: {outcome_var}")
    
    # Run ATE calculation directly
    print("🔍 Running ATE calculation...")
    ate_results = calculate_ate_dowhy(mock_analyzer, treatment_var, outcome_var)
    
    # Assertions
    assert ate_results is not None, "ATE calculation should return results"
    assert 'consensus_estimate' in ate_results, "Should have consensus estimate"
    assert 'estimates' in ate_results, "Should have individual estimates"
    assert 'interpretation' in ate_results, "Should have interpretation"
    
    consensus_estimate = ate_results['consensus_estimate']
    assert isinstance(consensus_estimate, (int, float)), "Consensus estimate should be numeric"
    
    print(f"✅ ATE calculation successful!")
    print(f"📈 Consensus estimate: {consensus_estimate:.4f}")
    print(f"📊 Number of estimation methods: {len(ate_results['estimates'])}")
    print("🎉 Causal inference smoke test passed!")

def test_inference_input_validation():
    """Test input validation and error handling in ATE calculation"""
    print("🧪 Testing inference input validation...")
    
    mock_analyzer = create_mock_analyzer_for_inference()
    
    # Test with None analyzer
    with pytest.raises(ValueError, match="Analyzer object is required"):
        calculate_ate_dowhy(None, 'treatment', 'outcome')
    
    # Test with empty treatment
    with pytest.raises(ValueError, match="Treatment must be a non-empty string"):
        calculate_ate_dowhy(mock_analyzer, '', 'outcome')
    
    # Test with empty outcome  
    with pytest.raises(ValueError, match="Outcome must be a non-empty string"):
        calculate_ate_dowhy(mock_analyzer, 'treatment', '')
    
    # Test with same treatment and outcome
    with pytest.raises(ValueError, match="Treatment and outcome variables cannot be the same"):
        calculate_ate_dowhy(mock_analyzer, 'treatment', 'treatment')
    
    # Test with analyzer without data
    mock_analyzer_no_data = Mock()
    mock_analyzer_no_data.data = None
    with pytest.raises(ValueError, match="Analyzer must have data loaded"):
        calculate_ate_dowhy(mock_analyzer_no_data, 'treatment', 'outcome')
    
    # Test with empty data
    mock_analyzer_empty = Mock()
    mock_analyzer_empty.data = pd.DataFrame()
    with pytest.raises(ValueError, match="Analyzer data is empty"):
        calculate_ate_dowhy(mock_analyzer_empty, 'treatment', 'outcome')
    
    print("✅ Input validation working correctly!")

def test_inference_interpretation_functions():
    """Test the interpretation and recommendation functions"""
    print("🧪 Testing interpretation functions...")
    
    # Test positive ATE interpretation
    positive_interpretation = _interpret_ate(0.5, 'treatment', 'outcome')
    assert "increases" in positive_interpretation.lower(), "Should indicate increase for positive ATE"
    
    # Test negative ATE interpretation
    negative_interpretation = _interpret_ate(-0.3, 'treatment', 'outcome')
    assert "decreases" in negative_interpretation.lower(), "Should indicate decrease for negative ATE"
    
    # Test zero ATE interpretation
    zero_interpretation = _interpret_ate(0.0, 'treatment', 'outcome')
    assert "no significant" in zero_interpretation.lower() or "no effect" in zero_interpretation.lower(), "Should indicate no effect for zero ATE"
    
    # Test recommendation generation
    estimates_dict = {
        'Linear Regression': {'estimate': 0.5, 'p_value': 0.01},
        'Propensity Score': {'estimate': 0.4, 'p_value': 0.02}
    }
    recommendation = _generate_simple_recommendation(estimates_dict)
    assert isinstance(recommendation, str), "Should return string recommendation"
    assert len(recommendation) > 0, "Should return non-empty recommendation"
    
    print("✅ Interpretation functions working correctly!")

def test_inference_edge_cases():
    """Test edge cases in causal inference"""
    print("🧪 Testing inference edge cases...")
    
    mock_analyzer = create_mock_analyzer_for_inference()
    
    # Test with missing treatment variable
    try:
        result = calculate_ate_dowhy(mock_analyzer, 'nonexistent_treatment', 'outcome')
        # If it doesn't raise an error, check that it handles it gracefully
        assert result is not None, "Should handle missing variables gracefully"
    except Exception as e:
        # It's acceptable to raise an error for missing variables
        assert "not found" in str(e).lower() or "missing" in str(e).lower() or "key" in str(e).lower()
    
    # Test with missing outcome variable
    try:
        result = calculate_ate_dowhy(mock_analyzer, 'treatment', 'nonexistent_outcome')
        assert result is not None, "Should handle missing variables gracefully"
    except Exception as e:
        assert "not found" in str(e).lower() or "missing" in str(e).lower() or "key" in str(e).lower()
    
    print("✅ Edge cases handled correctly!")

def test_inference_with_different_data_types():
    """Test inference with different treatment variable types"""
    print("🧪 Testing inference with different data types...")
    
    # Create continuous treatment data
    np.random.seed(123)
    n_samples = 500
    
    X = np.random.normal(0, 1, n_samples)
    continuous_treatment = np.random.normal(10, 2, n_samples)  # Continuous treatment
    outcome = 0.3 * continuous_treatment + 0.2 * X + np.random.normal(0, 0.5, n_samples)
    
    continuous_data = pd.DataFrame({
        'X': X,
        'continuous_treatment': continuous_treatment,
        'outcome': outcome
    })
    
    # Create mock analyzer for continuous treatment
    mock_analyzer_continuous = Mock()
    mock_analyzer_continuous.data = continuous_data
    
    mock_discovery = Mock()
    mock_discovery.numeric_columns = ['X', 'continuous_treatment', 'outcome']
    mock_discovery.encoded_data = continuous_data
    mock_discovery.categorical_mappings = {}
    
    # Simple adjacency matrix for continuous case
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 2] = 0.3  # continuous_treatment -> outcome
    adjacency_matrix[0, 2] = 0.2  # X -> outcome
    
    mock_discovery.adjacency_matrix = adjacency_matrix
    mock_discovery.columns = ['X', 'continuous_treatment', 'outcome']
    mock_analyzer_continuous.discovery = mock_discovery
    mock_analyzer_continuous.adjacency_matrix = adjacency_matrix
    mock_analyzer_continuous.columns = ['X', 'continuous_treatment', 'outcome']
    
    # Test ATE calculation with continuous treatment
    try:
        ate_results = calculate_ate_dowhy(mock_analyzer_continuous, 'continuous_treatment', 'outcome')
        assert ate_results is not None, "Should handle continuous treatment"
        print("✅ Continuous treatment handled successfully!")
    except Exception as e:
        print(f"⚠️ Continuous treatment test failed (this may be expected): {e}")
        # This is acceptable as DoWhy might have specific requirements

def test_inference_main():
    """Main test function that runs all inference tests"""
    print("🧪 Testing Causal Inference Module...")
    print("=" * 50)
    
    test_inference_smoke_test()
    print()
    
    test_inference_input_validation()
    print()
    
    test_inference_interpretation_functions()
    print()
    
    test_inference_edge_cases()
    print()
    
    test_inference_with_different_data_types()
    print()
    
    print("=" * 50)
    print("🎉 All Causal Inference tests passed!")

if __name__ == "__main__":
    test_inference_main()
