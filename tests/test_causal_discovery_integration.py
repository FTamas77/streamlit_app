#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from causal.analyzer import CausalAnalyzer

def test_causal_discovery_smoke_test():
    """
    Smoke test: Basic causal discovery without constraints (happy path).
    This is a simple integration test to verify the core functionality works.
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
    
    # Initialize analyzer
    analyzer = CausalAnalyzer() 
    analyzer.data = data
    
    # Run discovery without constraints (smoke test)
    print("🔍 Running causal discovery...")
    success = analyzer.run_causal_discovery()
    
    # Assertions
    assert success, "Basic causal discovery should succeed"
    assert analyzer.adjacency_matrix is not None, "Adjacency matrix should be generated"
    assert analyzer.adjacency_matrix.shape == (3, 3), "Adjacency matrix should be 3x3"
    
    # Check that some causal relationships were discovered
    total_edges = np.sum(np.abs(analyzer.adjacency_matrix) > 0.01)
    assert total_edges > 0, "Should discover at least some causal relationships"
    
    print("✅ Smoke test passed: Basic causal discovery works")

def test_causal_discovery_with_constraints():
    """
    Test causal discovery with prior knowledge - structure aligned with LLM.
    Tests the constraint format that matches what the LLM module generates.
    """
    print("🧪 Running causal discovery with constraints test...")
    
    np.random.seed(44)
    n_samples = 1000
    
    # Create causal structure: A -> B -> C, A -> C
    A = np.random.normal(0, 1, n_samples)
    B = 1.5 * A + np.random.normal(0, 0.3, n_samples)  
    C = 0.8 * A + 1.2 * B + np.random.normal(0, 0.3, n_samples)
    
    data = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C
    })
    
    print(f"📊 Created constrained test data with shape: {data.shape}")
      # Define constraints in LLM-aligned structure
    constraints = {
        "forbidden_edges": [["C", "A"], ["C", "B"]],  # C cannot cause A or B
        "required_edges": []
    }
    
    print(f"🚫 Using constraints: {constraints}")
    
    # Initialize analyzer with constraints
    analyzer = CausalAnalyzer()
    analyzer.data = data
    
    # Run discovery with constraints
    print("🔍 Running constrained causal discovery...")
    success = analyzer.run_causal_discovery(constraints=constraints)
    
    # Assertions
    assert success, "Constrained causal discovery should succeed"
    assert analyzer.adjacency_matrix is not None, "Adjacency matrix should be generated"
    
    print("✅ Constraint test passed: Causal discovery with constraints works")

if __name__ == "__main__":
    """Manual execution for debugging."""
    
    print("=" * 70)
    print("🧪 CAUSAL DISCOVERY INTEGRATION TESTS - MANUAL EXECUTION")
    print("=" * 70)
    
    tests = [
        ("Smoke Test (No Constraints)", test_causal_discovery_smoke_test),
        ("Prior Knowledge Constraints", test_causal_discovery_with_constraints),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🎯 {test_name}")
        print(f"{'='*50}")
        
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
    
    print(f"\n{'='*70}")
    print(f"📊 FINAL RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    
    if failed == 0:
        print("🎉 All tests passed! Causal discovery integration is working!")
    else:
        print("⚠️ Some tests failed - check the output above for details")
