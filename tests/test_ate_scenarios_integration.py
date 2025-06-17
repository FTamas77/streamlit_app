#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from causal.analyzer import CausalAnalyzer

def test_ate_scenarios_integration():
    """Test that policy scenarios are properly integrated into ATE calculation"""
    # Create sample data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'treatment': np.random.binomial(1, 0.5, n),
        'confounder': np.random.normal(0, 1, n),
        'outcome': np.random.normal(0, 1, n)
    })
    
    # Make the outcome somewhat dependent on treatment for realistic results
    data['outcome'] = data['outcome'] + 0.5 * data['treatment'] + 0.3 * data['confounder']
      # Initialize analyzer
    analyzer = CausalAnalyzer()
    analyzer.data = data  # Direct assignment instead of load_data for testing
      # Run causal discovery (basic step)
    analyzer.run_causal_discovery()    
    # Calculate ATE - this should now include core ATE analysis
    results = analyzer.calculate_ate('treatment', 'outcome', ['confounder'])
    
    # Check that core ATE results are present
    assert 'estimates' in results, "ATE estimates not found in results"
    assert 'consensus_estimate' in results, "Consensus estimate not found in results"
    assert 'interpretation' in results, "Interpretation not found in results"
    assert 'recommendation' in results, "Recommendation not found in results"
    
    # Verify the consensus estimate is reasonable
    ate_estimate = results['consensus_estimate']
    assert isinstance(ate_estimate, (int, float)), "ATE estimate should be numeric"
    
    # Check estimates structure
    estimates = results['estimates']
    assert len(estimates) > 0, "No ATE estimation methods found"

    print("✅ ATE integration test passed!")
    print(f"📊 ATE estimate: {results['consensus_estimate']:.4f}")
    print(f"🔍 Methods used: {list(estimates.keys())}")

if __name__ == "__main__":
    test_ate_scenarios_integration()
