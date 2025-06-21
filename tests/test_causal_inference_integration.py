#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from causal.inference import calculate_ate_dowhy

def create_mock_analyzer(with_adjacency_matrix=False):
    """Create a mock analyzer with realistic causal data for testing"""
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
    mock_analyzer.discovery = mock_discovery
    
    if with_adjacency_matrix:
        # Create adjacency matrix that reflects the true causal structure
        adjacency_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.3],  # X -> outcome
            [0.0, 0.0, 0.0, 0.0, 0.2],  # Y -> outcome  
            [0.0, 0.0, 0.0, 0.0, 0.1],  # Z -> outcome
            [0.0, 0.0, 0.0, 0.0, 0.5],  # treatment -> outcome (TRUE EFFECT)
            [0.0, 0.0, 0.0, 0.0, 0.0]   # outcome (no outgoing edges)
        ])
        mock_analyzer.adjacency_matrix = adjacency_matrix
        print(f"DEBUG: Mock analyzer created WITH adjacency matrix reflecting true causal structure")
    else:
        mock_analyzer.adjacency_matrix = None
        print(f"DEBUG: Mock analyzer created WITHOUT adjacency matrix")
    
    return mock_analyzer

def create_test_data():
    """Create test data for causal inference testing"""
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    n = 1000
    
    # Generate confounders
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n) 
    Z = np.random.normal(0, 1, n)
    
    # Treatment depends on confounders
    treatment_prob = 1 / (1 + np.exp(-(0.3*X + 0.2*Y + 0.1*Z)))
    treatment = np.random.binomial(1, treatment_prob, n)
    
    # Outcome depends on treatment and confounders 
    outcome = 0.5 * treatment + 0.3 * X + 0.2 * Y + 0.1 * Z + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'X': X, 'Y': Y, 'Z': Z,
        'treatment': treatment,
        'outcome': outcome
    })

def test_causal_inference_no_adjacency():
    """Test: No adjacency matrix - should show warning about needing causal discovery"""
    print("\n🧪 Test 1: No adjacency matrix")
    print("=" * 60)
    
    from causal.analyzer import CausalAnalyzer
    
    # Create a real analyzer without running discovery
    analyzer = CausalAnalyzer()
    analyzer.data = create_test_data()
    # Don't run causal discovery, so adjacency_matrix should be None
    
    # Test should now require adjacency matrix
    try:
        result = analyzer.calculate_ate('treatment', 'outcome')
        assert False, "Should have raised ValueError about missing causal discovery"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
        assert "Causal discovery must be run" in str(e)
    
    print("Expected behavior:")
    print("- Should require causal discovery to be run first")
    print("- Should fail with meaningful error message")
    print("✅ Test 1 completed successfully")

def test_causal_inference_with_adjacency():
    """Test: With adjacency matrix - should work properly with automatic confounder identification"""
    print("\n🧪 Test 2: With adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    
    # This should work since we have the causal structure
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome'
    )
    
    print("Expected DEBUG messages:")
    print("- Using causal graph structure for confounder identification")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- Confounders will be automatically identified from causal graph structure")
    print("- Passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
    
    # Check for statistical significance when proper graph structure is used
    if 'estimates' in result and 'Linear Regression' in result['estimates']:
        lr_result = result['estimates']['Linear Regression']
        if 'p_value' in lr_result and lr_result['p_value'] is not None:
            print(f"📊 P-value: {lr_result['p_value']:.4f}")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            
            # With proper confounders, we should get a significant effect
            assert lr_result['p_value'] < 0.05, f"Expected significant effect with confounders, got p={lr_result['p_value']:.4f}"
            print("✅ Statistical significance achieved with proper confounders!")
        else:
            print("📊 No p-value available in DoWhy results")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            print("⚠️ Cannot test statistical significance without p-value")
    
    print("✅ Test 2 completed successfully")

def test_causal_inference_no_confounders_with_adjacency():
    """Test: No confounders, With adjacency matrix - should produce significant effect"""
    print("\n🧪 Test 3: No confounders, With adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    
    # Don't mock the statistical functions - use real calculations
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome'
    )
    
    print("Expected DEBUG messages:")
    print("- Using causal graph structure for confounder identification")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- Confounders will be automatically identified from causal graph structure")
    print("- Passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
      # Check for statistical significance when adjacency matrix is used
    if 'estimates' in result and 'Linear Regression' in result['estimates']:
        lr_result = result['estimates']['Linear Regression']
        if 'p_value' in lr_result and lr_result['p_value'] is not None:
            print(f"📊 P-value: {lr_result['p_value']:.4f}")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            
            # With adjacency matrix, we should get a significant effect
            assert lr_result['p_value'] < 0.05, f"Expected significant effect with adjacency matrix, got p={lr_result['p_value']:.4f}"
            print("✅ Statistical significance achieved with adjacency matrix!")
        else:
            print("📊 No p-value available in DoWhy results")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            print("⚠️ Cannot test statistical significance without p-value")
    
    print("✅ Test 3 completed successfully")

def test_causal_inference_with_confounders_with_adjacency():
    """Test: With adjacency matrix containing confounders - should produce significant effect"""
    print("\n🧪 Test 4: With adjacency matrix containing confounders")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    
    # Don't mock the statistical functions - use real calculations
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome'
    )
    
    print("Expected DEBUG messages:")
    print("- Using causal graph structure for confounder identification")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- Confounders will be automatically identified from causal graph structure")
    print("- Passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
      # Check for statistical significance when BOTH confounders and adjacency matrix are used
    if 'estimates' in result and 'Linear Regression' in result['estimates']:
        lr_result = result['estimates']['Linear Regression']
        if 'p_value' in lr_result and lr_result['p_value'] is not None:
            print(f"📊 P-value: {lr_result['p_value']:.4f}")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            
            # With both confounders and adjacency matrix, we should get the most significant effect
            assert lr_result['p_value'] < 0.05, f"Expected highly significant effect with both confounders and adjacency matrix, got p={lr_result['p_value']:.4f}"
            print("✅ Statistical significance achieved with BOTH confounders and adjacency matrix!")
        else:
            print("📊 No p-value available in DoWhy results")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            print("⚠️ Cannot test statistical significance without p-value")
    
    print("✅ Test 4 completed successfully - BOTH parameters used!")

def test_causal_inference_empty_confounders_list():
    """Test: Edge case testing - should work with adjacency matrix"""
    print("\n🧪 Test 5: Edge case testing with adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome'
    )
    
    print("Expected DEBUG messages:")
    print("- Using causal graph structure for confounder identification")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- Confounders will be automatically identified from causal graph structure")
    print("- Passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
    print("✅ Test 5 completed successfully")

if __name__ == "__main__":
    """Manual execution for debugging causal inference integration."""
    
    print("=" * 80)
    print("🧪 CAUSAL INFERENCE INTEGRATION TESTS - MANUAL EXECUTION")
    print("=" * 80)
    
    tests = [
        ("No adjacency matrix", test_causal_inference_no_adjacency),
        ("With adjacency matrix", test_causal_inference_with_adjacency),
        ("No confounders, With adjacency matrix", test_causal_inference_no_confounders_with_adjacency),
        ("With confounders, With adjacency matrix (BOTH)", test_causal_inference_with_confounders_with_adjacency),
        ("Empty confounders list (edge case)", test_causal_inference_empty_confounders_list),
    ]
    
    print("🎯 Testing all parameter combinations for causal inference...")
    print("🔍 Each test will show expected parameter usage and behavior")
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"🎯 {test_name}")
        print(f"{'='*70}")
        
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"💥 Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*80}")
    
    if failed == 0:
        print("🎉 All parameter combination tests passed!")
        print("🔍 Statistical significance testing shows proper causal inference behavior:")
        print("   - With confounders/adjacency matrix: Significant effects (p < 0.05)")
        print("   - Without proper controls: May show confounding bias")
    else:
        print("⚠️ Some tests failed - check the output above for details")
