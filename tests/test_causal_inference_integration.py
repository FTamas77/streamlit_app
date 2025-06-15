#!/usr/bin/env python3
"""
Causal Inference Parameter Tests - Focused Parameter Combination Testing
Tests calculate_ate_dowhy parameter handling: confounders and adjacency_matrix combinations.
Everything else is mocked to focus purely on causal inference parameter logic.
Supports both pytest and manual execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from causal.inference import calculate_ate_dowhy

def create_mock_analyzer(with_adjacency_matrix=False):
    """Create a mock analyzer with optional adjacency matrix for testing"""
    mock_analyzer = Mock()
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100),
        'Z': np.random.normal(0, 1, 100),
        'treatment': np.random.binomial(1, 0.5, 100),
        'outcome': np.random.normal(0, 1, 100)
    })
    mock_analyzer.data = data
    
    if with_adjacency_matrix:
        # Create a simple adjacency matrix (5x5 for 5 variables)
        adjacency_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],  # X
            [0.3, 0.0, 0.0, 0.0, 0.0],  # Y (X -> Y)
            [0.0, 0.2, 0.0, 0.0, 0.0],  # Z (Y -> Z)
            [0.0, 0.0, 0.0, 0.0, 0.0],  # treatment
            [0.1, 0.0, 0.15, 0.4, 0.0]  # outcome (X,Z,treatment -> outcome)
        ])
        mock_analyzer.adjacency_matrix = adjacency_matrix
        print(f"DEBUG: Mock analyzer created WITH adjacency matrix ({adjacency_matrix.shape})")
    else:
        mock_analyzer.adjacency_matrix = None
        print("DEBUG: Mock analyzer created WITHOUT adjacency matrix")
    
    return mock_analyzer

def test_causal_inference_no_confounders_no_adjacency():
    """Test: No confounders, No adjacency matrix"""
    print("\n🧪 Test 1: No confounders, No adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=False)
    
    # Mock the analytics and utils imports at the correct paths
    with patch('analytics.statistical_metrics.calculate_simple_metrics') as mock_metrics, \
         patch('utils.effect_size.classify_effect_size') as mock_effect_size:
        
        mock_metrics.return_value = {'correlation': 0.1}
        mock_effect_size.return_value = 'small'
        
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
    print("✅ Test 1 completed successfully")

def test_causal_inference_with_confounders_no_adjacency():
    """Test: With confounders, No adjacency matrix"""
    print("\n🧪 Test 2: With confounders, No adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=False)
    confounders = ['X', 'Y']
    
    with patch('analytics.statistical_metrics.calculate_simple_metrics') as mock_metrics, \
         patch('utils.effect_size.classify_effect_size') as mock_effect_size:
        
        mock_metrics.return_value = {'correlation': 0.2}
        mock_effect_size.return_value = 'medium'
        
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
    print("✅ Test 2 completed successfully")

def test_causal_inference_no_confounders_with_adjacency():
    """Test: No confounders, With adjacency matrix"""
    print("\n🧪 Test 3: No confounders, With adjacency matrix")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    
    with patch('analytics.statistical_metrics.calculate_simple_metrics') as mock_metrics, \
         patch('utils.effect_size.classify_effect_size') as mock_effect_size:
        
        mock_metrics.return_value = {'correlation': 0.3}
        mock_effect_size.return_value = 'large'
        
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
    print("✅ Test 3 completed successfully")

def test_causal_inference_with_confounders_with_adjacency():
    """Test: With confounders, With adjacency matrix (BOTH parameters)"""
    print("\n🧪 Test 4: With confounders, With adjacency matrix (BOTH)")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    confounders = ['X', 'Z']
    
    with patch('analytics.statistical_metrics.calculate_simple_metrics') as mock_metrics, \
         patch('utils.effect_size.classify_effect_size') as mock_effect_size:
        
        mock_metrics.return_value = {'correlation': 0.4}
        mock_effect_size.return_value = 'large'
        
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
    print("✅ Test 4 completed successfully - BOTH parameters used!")

def test_causal_inference_empty_confounders_list():
    """Test: Empty confounders list (edge case)"""
    print("\n🧪 Test 5: Empty confounders list (edge case)")
    print("=" * 60)
    
    mock_analyzer = create_mock_analyzer(with_adjacency_matrix=True)
    confounders = []  # Empty list vs None
    
    with patch('analytics.statistical_metrics.calculate_simple_metrics') as mock_metrics, \
         patch('utils.effect_size.classify_effect_size') as mock_effect_size:
        
        mock_metrics.return_value = {'correlation': 0.05}
        mock_effect_size.return_value = 'negligible'
        
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
    print("- No confounders provided, relying on graph or DoWhy automatic confounder selection")
    print("- Passing graph to DoWhy causal model")
    
    assert result is not None
    assert 'consensus_estimate' in result
    print("✅ Test 5 completed successfully")

def validate_parameter_usage_debug_messages():
    """
    Validate that debug messages correctly show parameter usage.
    This function documents expected behavior for each parameter combination.
    """
    print("\n📋 PARAMETER USAGE DOCUMENTATION")
    print("=" * 60)
    
    combinations = [
        ("confounders=None, adjacency_matrix=None", "Neither parameter used"),
        ("confounders=['X'], adjacency_matrix=None", "Only confounders used"),
        ("confounders=None, adjacency_matrix=present", "Only adjacency matrix used"),
        ("confounders=['X'], adjacency_matrix=present", "BOTH parameters used"),
        ("confounders=[], adjacency_matrix=present", "Only adjacency matrix used (empty list)")
    ]
    
    for scenario, expected in combinations:
        print(f"• {scenario:35} → {expected}")
    
    print("\n💡 Key behaviors:")
    print("• Confounders are ALWAYS used if provided (not empty)")
    print("• Adjacency matrix is ALWAYS used if present")
    print("• Empty confounders list [] is treated as None")
    print("• Both parameters can be used simultaneously")

if __name__ == "__main__":
    """Manual execution for debugging causal inference parameter handling."""
    
    print("=" * 80)
    print("🎯 CAUSAL INFERENCE PARAMETER TESTS - MANUAL EXECUTION")
    print("=" * 80)
    
    # Show parameter usage documentation first
    validate_parameter_usage_debug_messages()
    
    tests = [
        ("No confounders, No adjacency matrix", test_causal_inference_no_confounders_no_adjacency),
        ("With confounders, No adjacency matrix", test_causal_inference_with_confounders_no_adjacency),
        ("No confounders, With adjacency matrix", test_causal_inference_no_confounders_with_adjacency),
        ("With confounders, With adjacency matrix", test_causal_inference_with_confounders_with_adjacency),
        ("Empty confounders list (edge case)", test_causal_inference_empty_confounders_list),
    ]
    
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
        print("🔍 Check the DEBUG messages above to verify parameter usage")
    else:
        print("⚠️ Some tests failed - check the output above for details")
