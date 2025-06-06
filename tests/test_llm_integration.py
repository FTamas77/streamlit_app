# Simple integration tests to study LLM interface and functionality
# Run these tests with actual OpenAI API to see how everything works together

import pytest
import os
from ai.llm_integration import generate_domain_constraints, explain_results_with_llm

# Simple API key loading - no conftest.py needed
def get_api_key():
    """Get API key from local config or environment"""
    try:
        from tests.local_config import OPENAI_API_KEY
        return OPENAI_API_KEY
    except ImportError:
        return os.getenv('OPENAI_API_KEY')

class TestLLMIntegration:
    """Integration tests for LLM functionality - studying the interface"""
    
    def test_generate_constraints_integration(self):
        """Test constraint generation with real data to study the interface"""
        api_key = get_api_key()
        if not api_key:
            pytest.skip("No API key available for integration tests")
            
        # Sample data columns like a real dataset
        columns = ['age', 'income', 'education', 'purchase_amount']
        domain_context = "Customer purchase behavior data where age and education influence income, and income affects purchase decisions"
        
        # Call the actual function with API key
        result = generate_domain_constraints(columns, domain_context, api_key=api_key)
        
        # Study the interface - what does the function return?
        print(f"Generated constraints: {result}")
        
        # Basic structure validation
        assert isinstance(result, dict)
        assert 'forbidden_edges' in result
        assert 'required_edges' in result 
        assert 'temporal_order' in result
        assert 'explanation' in result
        
        # Study the data types
        assert isinstance(result['forbidden_edges'], list)
        assert isinstance(result['required_edges'], list)
        assert isinstance(result['temporal_order'], list)
        assert isinstance(result['explanation'], str)
    
    def test_explain_results_integration(self):
        """Test result explanation with real data to study the interface"""
        api_key = get_api_key()
        if not api_key:
            pytest.skip("No API key available for integration tests")
            
        # Sample analysis results like what causal inference would produce
        ate_results = {
            'consensus_estimate': 0.15,
            'estimates': {
                'linear_regression': {'estimate': 0.12, 'p_value': 0.03, 'confidence_interval': [0.02, 0.22]},
                'matching': {'estimate': 0.18, 'p_value': 0.01, 'confidence_interval': [0.08, 0.28]}
            },
            'recommendation': 'Results are consistent across methods',
            'interpretation': 'Moderate positive causal effect detected'
        }
        
        treatment = "marketing_campaign"
        outcome = "sales_increase"
        
        # Call the actual function with API key
        result = explain_results_with_llm(ate_results, treatment, outcome, api_key=api_key)
        
        # Study the interface - what does the function return?
        print(f"AI explanation: {result}")
        
        # Basic validation
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Study content - should contain business-relevant terms
        result_lower = result.lower()
        # These might appear in a good business explanation
        business_terms = ['effect', 'impact', 'recommendation', 'business', 'result']
        
        # At least some business terminology should appear
        found_terms = [term for term in business_terms if term in result_lower]
        assert len(found_terms) > 0, f"Expected business terms in explanation, got: {result}"
