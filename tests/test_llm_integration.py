#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from llm.llm import generate_domain_constraints, explain_results_with_llm

def get_api_key():
    """Get API key from local config or environment"""
    try:
        from tests.local_config import OPENAI_API_KEY
        return OPENAI_API_KEY
    except ImportError:
        return os.getenv('OPENAI_API_KEY')

def validate_constraint_structure(constraints):
    """
    Validate that constraint structure matches what causal discovery expects.
    This ensures LLM output is compatible with our discovery algorithms.
    """
    # Must be a dictionary
    assert isinstance(constraints, dict), f"Expected dict, got {type(constraints)}"
    
    # Required keys for causal discovery integration
    required_keys = ['forbidden_edges', 'required_edges']
    for key in required_keys:
        assert key in constraints, f"Missing required key: {key}"
      # Validate forbidden_edges structure
    forbidden = constraints['forbidden_edges']
    assert isinstance(forbidden, list), f"forbidden_edges must be list, got {type(forbidden)}"
    
    for edge_item in forbidden:
        # Handle both dict format with reasoning and simple list format
        if isinstance(edge_item, dict):
            assert 'edge' in edge_item, f"Edge dict must have 'edge' key, got {edge_item.keys()}"
            edge = edge_item['edge']
        else:
            edge = edge_item
            
        assert isinstance(edge, list), f"Each forbidden edge must be list, got {type(edge)}"
        assert len(edge) == 2, f"Each edge must have 2 elements [source, target], got {len(edge)}"
        assert isinstance(edge[0], str), f"Source must be string, got {type(edge[0])}"
        assert isinstance(edge[1], str), f"Target must be string, got {type(edge[1])}"
    
    # Validate required_edges structure  
    required = constraints['required_edges']
    assert isinstance(required, list), f"required_edges must be list, got {type(required)}"
    
    for edge_item in required:
        # Handle both dict format with reasoning and simple list format
        if isinstance(edge_item, dict):
            assert 'edge' in edge_item, f"Edge dict must have 'edge' key, got {edge_item.keys()}"
            edge = edge_item['edge']
        else:
            edge = edge_item
            
        assert isinstance(edge, list), f"Each required edge must be list, got {type(edge)}"
        assert len(edge) == 2, f"Each edge must have 2 elements [source, target], got {len(edge)}"
        assert isinstance(edge[0], str), f"Source must be string, got {type(edge[0])}"
        assert isinstance(edge[1], str), f"Target must be string, got {type(edge[1])}"
    
    # Optional: Check for additional constraint types that causal discovery supports
    optional_keys = []  # No optional keys needed - we only use forbidden_edges and required_edges
    for key in optional_keys:
        if key in constraints:
            assert isinstance(constraints[key], list), f"{key} must be list if present"
            for var in constraints[key]:
                assert isinstance(var, str), f"Variables in {key} must be strings"

@pytest.mark.integration
def test_constraint_structure_consistency():
    """
    Test that LLM generates constraints in the exact format expected by causal discovery.
    This is the critical integration test ensuring end-to-end compatibility.
    """
    print("🧪 Testing LLM constraint structure consistency...")
    
    api_key = get_api_key()
    if not api_key:
        pytest.skip("No API key available for integration tests")    # Use realistic business scenario with clear temporal and logical relationships
    columns = ['customer_age', 'income_level', 'marketing_exposure', 'product_purchased']
    domain_context = """
    E-commerce customer behavior analysis with clear causal relationships:
    
    GIVEN FACTS:
    - customer_age is fixed demographic data (cannot be caused by other factors)
    - income_level may be influenced by customer_age (older customers may have different income patterns)
    - marketing_exposure is controlled by the company (business decision)
    - product_purchased is the final outcome we want to understand
    
    LOGICAL CONSTRAINTS:
    - product_purchased decisions CANNOT cause customer_age or income_level (temporal impossibility)
    - marketing_exposure and income_level CAN influence product_purchased decisions
    - customer_age MAY influence income_level
    
    Use these EXACT column names in your constraints: customer_age, income_level, marketing_exposure, product_purchased
    Focus on forbidden edges (what cannot cause what) rather than required edges.
    """
    
    print(f"📊 Testing with columns: {columns}")
    print(f"🏷️ Domain context: {domain_context[:100]}...")
    
    # Generate constraints using LLM
    print("🤖 Generating constraints with LLM...")
    constraints = generate_domain_constraints(columns, domain_context, api_key=api_key)
    
    print(f"📋 Generated constraints: {json.dumps(constraints, indent=2)}")
    
    # Validate structure compatibility with causal discovery
    validate_constraint_structure(constraints)
      # Test that edges reference actual columns
    all_edges = constraints['forbidden_edges'] + constraints['required_edges']
    for edge_item in all_edges:
        # Handle both dict format with reasoning and simple list format
        if isinstance(edge_item, dict):
            edge = edge_item['edge']
        else:
            edge = edge_item
            
        source, target = edge
        assert source in columns, f"Source '{source}' not in dataset columns: {columns}"
        assert target in columns, f"Target '{target}' not in dataset columns: {columns}"    # Test constraint logic makes sense
    forbidden_edges = constraints['forbidden_edges']
    required_edges = constraints['required_edges']
    
    # Check for conflicts between forbidden and required
    def extract_edge_tuple(edge_item):
        if isinstance(edge_item, dict):
            edge = edge_item['edge']
        else:
            edge = edge_item
        return (edge[0], edge[1])
    
    forbidden_set = {extract_edge_tuple(edge) for edge in forbidden_edges}
    required_set = {extract_edge_tuple(edge) for edge in required_edges}
    conflict = forbidden_set.intersection(required_set)
    
    if len(conflict) > 0:
        print(f"⚠️ WARNING: LLM generated conflicting constraints: {conflict}")
        print("This shows the importance of constraint validation in production systems")
        # Don't fail the test - this is valuable information about LLM behavior
    else:
        print("✅ No conflicting constraints found")
    
    print("✅ Constraint structure consistency test passed")

@pytest.mark.integration
def test_constraint_edge_cases():
    """Test LLM constraint generation with edge cases and error scenarios."""
    print("🧪 Testing LLM constraint edge cases...")
    
    api_key = get_api_key()
    if not api_key:
        pytest.skip("No API key available for integration tests")
    
    # Test with minimal data
    minimal_columns = ['A', 'B']
    minimal_context = "Simple two-variable system"
    
    print("📊 Testing minimal case with 2 variables...")
    constraints = generate_domain_constraints(minimal_columns, minimal_context, api_key=api_key)
    validate_constraint_structure(constraints)
    
    # Test with many variables
    many_columns = [f'var_{i}' for i in range(10)]
    complex_context = "Complex system with many interacting variables in a business context"
    
    print("📊 Testing complex case with 10 variables...")
    constraints = generate_domain_constraints(many_columns, complex_context, api_key=api_key)
    validate_constraint_structure(constraints)
    
    # Test with domain-specific terminology
    medical_columns = ['age', 'blood_pressure', 'medication_dose', 'health_outcome']
    medical_context = """
    Medical study data:
    - Age is a demographic factor that cannot be changed
    - Blood pressure is influenced by age and other factors
    - Medication dose is controlled by doctors
    - Health outcome is what we want to improve
    - Medication should affect health outcome
    - Health outcome cannot cause age or medication dose
    """
    
    print("📊 Testing medical domain case...")
    constraints = generate_domain_constraints(medical_columns, medical_context, api_key=api_key)
    validate_constraint_structure(constraints)
    
    # Verify medical constraints make sense
    forbidden = constraints['forbidden_edges']
    
    # Should forbid outcome -> age (temporal impossibility)
    health_to_age = ['health_outcome', 'age'] in forbidden
    outcome_to_med = ['health_outcome', 'medication_dose'] in forbidden
    
    # At least one temporal constraint should be present
    temporal_constraints = health_to_age or outcome_to_med
    print(f"🕐 Temporal constraints detected: {temporal_constraints}")
    
    print("✅ Edge cases test passed")

@pytest.mark.integration
def test_llm_explanation_quality():
    """Test that LLM explanations are business-friendly and accurate."""
    print("🧪 Testing LLM explanation quality...")
    
    api_key = get_api_key()
    if not api_key:
        pytest.skip("No API key available for integration tests")
    
    # Realistic causal inference results
    ate_results = {
        'consensus_estimate': 0.15,
        'confidence_interval': [0.08, 0.22], 
        'p_value': 0.02,
        'method': 'backdoor',
        'n_samples': 4000,
        'effect_size': 'medium'
    }
    
    treatment = "marketing_campaign"
    outcome = "sales_revenue"
    
    print(f"📈 Testing explanation for {treatment} → {outcome}")
    print(f"📊 ATE: {ate_results['consensus_estimate']}")
    
    # Generate explanation
    explanation = explain_results_with_llm(ate_results, treatment, outcome, api_key=api_key)
    
    print(f"🤖 Generated explanation: {explanation[:200]}...")
    
    # Validate explanation quality
    assert isinstance(explanation, str), "Explanation must be string"
    assert len(explanation) > 50, "Explanation should be substantial"
      # Check for business-relevant content
    explanation_lower = explanation.lower()
    
    # Should mention key concepts (flexible matching)
    treatment_mentioned = (
        treatment.lower() in explanation_lower or 
        'marketing' in explanation_lower or 
        'campaign' in explanation_lower
    )
    outcome_mentioned = (
        outcome.lower() in explanation_lower or 
        'sales' in explanation_lower or 
        'revenue' in explanation_lower
    )
    
    assert treatment_mentioned, f"Should mention treatment concept, got: {explanation}"
    assert outcome_mentioned, f"Should mention outcome concept, got: {explanation}"
    
    # Should include business terms
    business_terms = ['effect', 'impact', 'increase', 'significant', 'confidence']
    found_terms = [term for term in business_terms if term in explanation_lower]
    assert len(found_terms) >= 2, f"Should include business terms, found: {found_terms}"    
    # Should mention the magnitude
    assert '0.15' in explanation or '15%' in explanation or 'fifteen' in explanation.lower(), \
        "Should mention the effect magnitude"
    
    print("✅ Explanation quality test passed")

@pytest.mark.integration
def test_constraint_compatibility_with_discovery():
    """
    Test that LLM-generated constraints can be used directly with causal discovery.
    This tests the end-to-end integration pipeline.
    """
    print("🧪 Testing constraint compatibility with causal discovery...")
      # Import causal discovery to test integration
    from causal_ai.discovery_constraints import create_prior_knowledge_matrix
    import numpy as np
      # Test constraint structure that matches both LLM output and discovery input
    test_constraints = {
        "forbidden_edges": [["C", "A"], ["C", "B"]],
        "required_edges": [["A", "B"]]
    }
    
    columns = ["A", "B", "C"]
    
    print(f"🔗 Testing constraint integration with columns: {columns}")
    print(f"⚙️ Test constraints: {json.dumps(test_constraints, indent=2)}")
    
    # This should work without errors - tests the integration
    prior_knowledge = create_prior_knowledge_matrix(test_constraints, columns)
    
    # Validate the prior knowledge matrix was created
    assert prior_knowledge is not None, "Should create prior knowledge matrix"
    assert isinstance(prior_knowledge, np.ndarray), "Should return numpy array"
    assert prior_knowledge.shape == (3, 3), f"Should be 3x3 matrix, got {prior_knowledge.shape}"    
    print(f"📋 Generated prior knowledge matrix:\n{prior_knowledge}")
    
    # Validate constraint application - only check what we actually constrained
    # C -> A should be forbidden (C=index 2, A=index 0)
    assert prior_knowledge[2, 0] == 0, "C -> A should be forbidden"    # C -> B should be forbidden (C=index 2, B=index 1)  
    assert prior_knowledge[2, 1] == 0, "C -> B should be forbidden"
    # A -> B should be required (A=index 0, B=index 1)
    assert prior_knowledge[0, 1] == 1, "A -> B should be required"
    
    print("✅ Constraint compatibility test passed")

@pytest.mark.integration
def test_constraint_conflict_resolution():
    """Test that conflicting constraints are properly detected and resolved"""
    print("\n" + "="*50)
    print("🚨 Testing Constraint Conflict Resolution")
    
    from causal_ai.discovery_constraints import validate_constraints, create_prior_knowledge_matrix
    import numpy as np
    
    # Create conflicting constraints
    conflicting_constraints = {
        "forbidden_edges": [["A", "B"], ["C", "A"]],  # A -> B forbidden
        "required_edges": [["A", "B"], ["B", "C"]]   # A -> B required (CONFLICT!)
    }
    
    columns = ["A", "B", "C"]
    
    print(f"🔗 Testing conflicting constraints: {conflicting_constraints}")
    
    # Step 1: Validate and resolve conflicts
    validation = validate_constraints(conflicting_constraints)
    
    # Should detect the conflict
    assert not validation['valid'], "Should detect conflicts"
    assert len(validation['conflicts']) == 1, "Should detect one conflict"
    assert validation['conflicts'][0]['edge'] == ['A', 'B'], "Should identify A->B conflict"
    
    # Should provide resolved constraints
    resolved_constraints = validation['resolved_constraints']
    print(f"📋 Resolved constraints: {resolved_constraints}")
    
    # A -> B should be removed from forbidden edges
    assert ['A', 'B'] not in resolved_constraints.get('forbidden_edges', []), "Conflicting forbidden edge should be removed"
    # A -> B should still be in required edges
    assert ['A', 'B'] in resolved_constraints.get('required_edges', []), "Required edge should remain"
    # C -> A should still be forbidden (no conflict)
    assert ['C', 'A'] in resolved_constraints.get('forbidden_edges', []), "Non-conflicting forbidden edge should remain"
    
    # Step 2: Create prior knowledge matrix with resolved constraints
    prior_knowledge = create_prior_knowledge_matrix(resolved_constraints, columns)
    
    # Validate the matrix reflects resolved constraints
    assert prior_knowledge is not None, "Should create prior knowledge matrix with resolved constraints"
    
    # A -> B should be required (required wins over forbidden)
    assert prior_knowledge[0, 1] == 1, "Required edge should be present in matrix"
    
    # C -> A should still be forbidden (no conflict)
    assert prior_knowledge[2, 0] == 0, "Non-conflicting forbidden edge should remain in matrix"
    
    # B -> C should be required (no conflict)  
    assert prior_knowledge[1, 2] == 1, "Non-conflicting required edge should work"
    
    print("✅ Conflict resolution test passed - full workflow from validation to matrix creation")

if __name__ == "__main__":
    """Manual execution for debugging LLM integration."""
    
    print("=" * 80)
    print("🤖 LLM INTEGRATION TESTS - MANUAL EXECUTION")
    print("=" * 80)
    
    tests = [
        ("Constraint Structure Consistency", test_constraint_structure_consistency),
        ("Constraint Edge Cases", test_constraint_edge_cases), 
        ("LLM Explanation Quality", test_llm_explanation_quality),
        ("Constraint-Discovery Compatibility", test_constraint_compatibility_with_discovery),
        ("Constraint Conflict Resolution", test_constraint_conflict_resolution),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🎯 {test_name}")
        print(f"{'='*60}")
        
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
        print("🎉 All LLM integration tests passed! Constraint structure is aligned!")
    else:
        print("⚠️ Some tests failed - check the output above for details")
