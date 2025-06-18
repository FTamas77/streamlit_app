import pandas as pd
import numpy as np
import warnings
from typing import Dict, List

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

def _adjacency_to_dowhy_dot_graph(adjacency_matrix, columns, treatment: str, outcome: str, confounders: list = None):
    """Convert adjacency matrix to DOT graph format for DoWhy - properly interpret DirectLiNGAM matrix"""
    if adjacency_matrix is None or not hasattr(adjacency_matrix, 'shape'):
        print("DEBUG: adjacency_matrix is None or has no shape")
        return None
    
    print(f"DEBUG: Creating DOT graph for columns: {columns}")
    print(f"DEBUG: Treatment: {treatment}, Outcome: {outcome}, Confounders: {confounders}")
    print(f"DEBUG: Adjacency matrix shape: {adjacency_matrix.shape}")
        
    dot_graph = 'digraph {\n'

    # Declare nodes with valid DOT syntax
    all_nodes = set([treatment, outcome] + (confounders or []))
    for node in all_nodes:
        dot_graph += f'"{node}";\n'

    # Process adjacency matrix - DirectLiNGAM produces lower triangular matrices
    # where entry (i,j) represents the direct effect from variable j to variable i
    if adjacency_matrix.shape[0] == len(columns) and adjacency_matrix.shape[1] == len(columns):
        edges_added = 0
        
        for i, to_var in enumerate(columns):
            for j, from_var in enumerate(columns):
                # DirectLiNGAM: adjacency_matrix[i,j] is effect from j to i
                if abs(adjacency_matrix[i, j]) > 0.01:
                    print(f"DEBUG: Adding edge: {from_var} -> {to_var} (weight: {adjacency_matrix[i, j]:.3f})")
                    dot_graph += f'"{from_var}" -> "{to_var}";\n'
                    edges_added += 1
        
        print(f"DEBUG: Total edges added: {edges_added}")
    else:
        print(f"DEBUG: Matrix dimensions don't match columns. Matrix: {adjacency_matrix.shape}, Columns: {len(columns)}")

    dot_graph += '}'
    
    print(f"DEBUG: Generated DOT graph:\n{dot_graph}")
    print("=" * 50)
    
    return dot_graph

def _generate_simple_recommendation(estimates: Dict) -> str:
    """Generate simplified recommendation for speed"""
    if not estimates:
        return "❌ No reliable estimates available. Consider improving data quality."
    
    estimate_value = list(estimates.values())[0]["estimate"]
    p_value = list(estimates.values())[0].get("p_value")
    
    if p_value and p_value < 0.05:
        if abs(estimate_value) > 0.1:
            return "✅ Strong significant causal effect detected. Consider implementing interventions."
        else:
            return "📊 Statistically significant but small effect. Consider cost-benefit analysis."
    else:
        return "⚠️ No statistically significant causal effect detected. Explore other variables or collect more data."

def _interpret_ate(ate_value: float, treatment: str, outcome: str) -> str:
    """Generate interpretation of ATE"""
    if ate_value is None:
        return "Unable to determine causal effect"
    
    if ate_value > 0:
        direction = "increases"
    elif ate_value < 0:
        direction = "decreases"
    else:
        direction = "has no effect on"
        
    return f"A one-unit increase in {treatment} {direction} {outcome} by {abs(ate_value):.4f} units on average."

def calculate_ate_dowhy(analyzer, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
    """Calculate ATE using DoWhy - always uses confounders if provided, always uses adjacency matrix if present, with debug messages"""
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for ATE calculation but not available")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        from analytics.statistical_metrics import calculate_simple_metrics
        from utils.effect_size import classify_effect_size
        
        # Debug: show what parameters are present
        print(f"DEBUG: calculate_ate_dowhy called with treatment={treatment}, outcome={outcome}")
        print(f"DEBUG: confounders provided: {confounders}")
        print(f"DEBUG: analyzer has adjacency_matrix: {hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None}")
          # Only create DOT graph if we have an adjacency matrix from causal discovery
        graph = None
        if hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None:
            print("DEBUG: Using adjacency_matrix to build graph for DoWhy")
            
            # Use numeric columns from discovery if available
            if hasattr(analyzer.discovery, 'numeric_columns') and analyzer.discovery.numeric_columns:
                graph_columns = analyzer.discovery.numeric_columns
                print(f"DEBUG: Using numeric columns for graph: {graph_columns}")
            else:
                # Fallback to all columns (for backward compatibility)
                graph_columns = analyzer.data.columns.tolist()
                print(f"DEBUG: Using all columns for graph (fallback): {graph_columns}")
            
            dot_graph = _adjacency_to_dowhy_dot_graph(
                analyzer.adjacency_matrix, 
                graph_columns, 
                treatment, 
                outcome, 
                confounders
            )
            graph = dot_graph
        else:
            print("DEBUG: No adjacency_matrix available, not passing graph to DoWhy")
        
        # Always use confounders if provided
        if confounders:
            print(f"DEBUG: Using user-specified confounders: {confounders}")
        else:
            print("DEBUG: No confounders provided, relying on graph or DoWhy automatic confounder selection")
        
        # Create causal model - only pass graph if we have one
        causal_model_args = {
            'data': analyzer.data,
            'treatment': treatment,
            'outcome': outcome,
            'common_causes': confounders if confounders else None
        }
        
        if graph is not None:
            causal_model_args['graph'] = graph
            print("DEBUG: Passing graph to DoWhy causal model")
        else:
            print("DEBUG: Not passing graph to DoWhy causal model")
        
        causal_model = CausalModel(**causal_model_args)
        
        # Identify causal effect
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate the effect using linear regression
        causal_estimate = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            effect_modifiers=[]
        )
        
        # Extract confidence intervals and p-value
        try:
            confidence_interval = causal_estimate.get_confidence_intervals()
            # Handle nested numpy array format [[lower, upper]]
            if confidence_interval is not None and len(confidence_interval) > 0:
                if len(confidence_interval[0]) == 2:  # It's [[lower, upper]]
                    confidence_interval = [float(confidence_interval[0][0]), float(confidence_interval[0][1])]
                else:
                    confidence_interval = [None, None]
            else:
                confidence_interval = [None, None]
        except Exception as e:
            confidence_interval = [None, None]
        
        try:
            # Get p-value using test_stat_significance
            sig_results = causal_estimate.test_stat_significance()
            if isinstance(sig_results, dict) and 'p_value' in sig_results:
                p_value = float(sig_results['p_value'])
            else:
                p_value = None
        except Exception as e:
            p_value = None
          # Package results
        results = {
            "Linear Regression": {                "estimate": causal_estimate.value,                "confidence_interval": confidence_interval,
                "p_value": p_value,
                "method": "backdoor.linear_regression"
            }
        }
        
        return {
            "estimates": results,
            "consensus_estimate": causal_estimate.value,
            "robustness": {"status": "skipped_for_speed"},
            "interpretation": _interpret_ate(causal_estimate.value, treatment, outcome),
            "recommendation": _generate_simple_recommendation(results),
            "additional_metrics": calculate_simple_metrics(analyzer.data, treatment, outcome, causal_estimate.value)
        }
