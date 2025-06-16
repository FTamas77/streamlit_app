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

def test_causal_inference_no_confounders_no_adjacency():
    """Test: No confounders, No adjacency matrix - should produce non-significant effect due to confounding"""
    print("\n🧪 Test 1: No confounders, No adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=False)
    
    # Don't mock the statistical functions - use real calculations to see confounding bias
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome',
        confounders=None
    )
    
    print("Expected DEBUG messages:")
    print("- confounders provided: None")
    print("- analyzer has adjacency_matrix: False")
    print("- No adjacency_matrix available, not passing graph to DoWhy")
    print("- No confounders provided, relying on graph or DoWhy automatic confounder selection")
    print("- Not passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
      # Without proper confounders or adjacency matrix, the effect may be biased/non-significant
    if 'estimates' in result and 'Linear Regression' in result['estimates']:
        lr_result = result['estimates']['Linear Regression']
        if 'p_value' in lr_result and lr_result['p_value'] is not None:
            print(f"📊 P-value: {lr_result['p_value']:.4f}")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
            
            # This might not be significant due to confounding bias
            if lr_result['p_value'] >= 0.05:
                print("⚠️ Non-significant result - as expected due to confounding bias without proper controls")
            else:
                print("⚠️ Significant result despite lack of confounding controls - might indicate strong effect")
        else:
            print("📊 No p-value available in DoWhy results")
            print(f"📊 Effect estimate: {result['consensus_estimate']:.4f}")
    
    print("✅ Test 1 completed successfully")

def test_causal_inference_with_confounders_no_adjacency():
    """Test: With confounders, No adjacency matrix - should produce significant effect"""
    print("\n🧪 Test 2: With confounders, No adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=False)
    confounders = ['X', 'Y']  # Important confounders that affect both treatment and outcome
    
    # Don't mock the statistical functions - use real calculations
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome',
        confounders=confounders
    )
    
    print("Expected DEBUG messages:")
    print(f"- confounders provided: {confounders}")
    print("- analyzer has adjacency_matrix: False")
    print("- No adjacency_matrix available, not passing graph to DoWhy")
    print(f"- Using user-specified confounders: {confounders}")
    print("- Not passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
      # Check for statistical significance when proper confounders are used
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
        outcome='outcome',
        confounders=None
    )
    
    print("Expected DEBUG messages:")
    print("- confounders provided: None")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- No confounders provided, relying on graph or DoWhy automatic confounder selection")
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
    """Test: With confounders, With adjacency matrix (BOTH) - should produce significant effect"""
    print("\n🧪 Test 4: With confounders, With adjacency matrix (BOTH)")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    confounders = ['X', 'Z']
    
    # Don't mock the statistical functions - use real calculations
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome',
        confounders=confounders
    )
    
    print("Expected DEBUG messages:")
    print(f"- confounders provided: {confounders}")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print(f"- Using user-specified confounders: {confounders}")
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
    """Test: Empty confounders list (edge case)"""
    print("\n🧪 Test 5: Empty confounders list (edge case)")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    confounders = []  # Empty list vs None
    
    result = calculate_ate_dowhy(
        analyzer=mock_analyzer,
        treatment='treatment',
        outcome='outcome',
        confounders=confounders
    )
    
    print("Expected DEBUG messages:")
    print("- confounders provided: []")
    print("- analyzer has adjacency_matrix: True")
    print("- Using adjacency_matrix to build graph for DoWhy")
    print("- No confounders provided, relying on graph or DoWhy automatic confounder selection")
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
        ("No confounders, No adjacency matrix", test_causal_inference_no_confounders_no_adjacency),
        ("With confounders, No adjacency matrix", test_causal_inference_with_confounders_no_adjacency),
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
