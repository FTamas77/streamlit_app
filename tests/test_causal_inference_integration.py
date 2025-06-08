# Integration tests for causal inference functionality
# Testing ATE calculation with synthetic datasets - simplified test

import pytest
import numpy as np
import pandas as pd
from models.causal_analyzer import CausalAnalyzer

@pytest.mark.integration
def test_causal_inference_ate_integration():
    """Test ATE calculation with synthetic dataset - basic functionality test"""
    
    # Generate simple dataset with known treatment effect
    np.random.seed(42)
    n_samples = 1000
    
    # Simple data generation
    age = np.random.normal(40, 10, n_samples)
    income = 30000 + age * 1000 
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Simple outcome with clear treatment effect
    true_ate = 5000
    outcome = 20000 + age * 500 + income * 0.1 + treatment * true_ate
    
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'marketing_campaign': treatment,
        'sales': outcome
    })
    
    # Test causal inference
    analyzer = CausalAnalyzer()
    analyzer.data = data
    
    # Simple adjacency matrix representing the causal structure
    analyzer.adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],  # age
        [0.8, 0.0, 0.0, 0.0],  # income (from age)
        [0.0, 0.0, 0.0, 0.0],  # treatment (independent)
        [0.5, 0.1, 0.7, 0.0]   # sales (from age, income, treatment)
    ])
    
    # Test ATE calculation
    ate_result = analyzer.calculate_ate(
        treatment='marketing_campaign',
        outcome='sales'
    )
    
    # Basic functionality tests
    assert ate_result is not None, "Should return ATE results"
    assert isinstance(ate_result, dict), "Should return dictionary"
    assert 'consensus_estimate' in ate_result, "Should contain consensus estimate"
    assert 'estimates' in ate_result, "Should contain estimates"
    assert 'interpretation' in ate_result, "Should contain interpretation"
    
    estimated_ate = ate_result['consensus_estimate']
    error_percentage = abs(estimated_ate - true_ate) / true_ate * 100
    
    # Check confidence intervals and p-values
    linear_reg_result = ate_result['estimates']['Linear Regression']
    confidence_interval = linear_reg_result['confidence_interval']
    p_value = linear_reg_result['p_value']
    
    # Reasonable tests for basic functionality
    assert abs(estimated_ate) > 4000, "ATE should be substantial"
    assert error_percentage < 15.0, f"ATE estimate should be reasonably close (error: {error_percentage:.2f}%)"
    
    # Test confidence intervals
    if confidence_interval[0] is not None:
        assert len(confidence_interval) == 2, "Should have lower and upper bounds"
        assert confidence_interval[0] < confidence_interval[1], "Lower bound should be less than upper bound"
        assert confidence_interval[0] <= estimated_ate <= confidence_interval[1], "Estimate should be within CI"
        print(f"✅ Confidence interval: [{confidence_interval[0]:.0f}, {confidence_interval[1]:.0f}]")
    
    # Test p-value
    if p_value is not None:
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"✅ P-value: {p_value:.4f} ({significance})")
    
    print(f"✅ ATE calculation test passed! Estimated: {estimated_ate:.0f}, True: {true_ate}, Error: {error_percentage:.1f}%")
    
    # Validate dataset structure
    assert data.shape[0] == n_samples
    assert 'marketing_campaign' in data.columns
    assert 'sales' in data.columns
