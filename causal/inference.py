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
        print(f"DEBUG: ================ CAUSAL INFERENCE DEBUG ================")
        print(f"DEBUG: calculate_ate_dowhy called with treatment={treatment}, outcome={outcome}")
        print(f"DEBUG: confounders provided: {confounders}")
        print(f"DEBUG: analyzer has adjacency_matrix: {hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None}")
        
        # Debug: Check data being used
        print(f"DEBUG: analyzer.data shape: {analyzer.data.shape}")
        print(f"DEBUG: analyzer.data columns: {list(analyzer.data.columns)}")
        print(f"DEBUG: Treatment '{treatment}' in analyzer.data: {treatment in analyzer.data.columns}")
        print(f"DEBUG: Outcome '{outcome}' in analyzer.data: {outcome in analyzer.data.columns}")
        
        # Check if we have encoded data available
        if hasattr(analyzer.discovery, 'encoded_data') and analyzer.discovery.encoded_data is not None:
            print(f"DEBUG: encoded_data available with shape: {analyzer.discovery.encoded_data.shape}")
            print(f"DEBUG: encoded_data columns: {list(analyzer.discovery.encoded_data.columns)}")
            print(f"DEBUG: Treatment '{treatment}' in encoded_data: {treatment in analyzer.discovery.encoded_data.columns}")
            print(f"DEBUG: Outcome '{outcome}' in encoded_data: {outcome in analyzer.discovery.encoded_data.columns}")
        else:
            print(f"DEBUG: No encoded_data available")
        
        # Check variable values and distributions
        if treatment in analyzer.data.columns:
            print(f"DEBUG: Treatment '{treatment}' stats in analyzer.data:")
            print(f"       - Unique values: {analyzer.data[treatment].nunique()}")
            print(f"       - Value range: [{analyzer.data[treatment].min():.3f}, {analyzer.data[treatment].max():.3f}]")
            print(f"       - Mean: {analyzer.data[treatment].mean():.3f}")
            print(f"       - Std: {analyzer.data[treatment].std():.3f}")
            print(f"       - Sample values: {list(analyzer.data[treatment].head())}")
        
        if outcome in analyzer.data.columns:
            print(f"DEBUG: Outcome '{outcome}' stats in analyzer.data:")
            print(f"       - Unique values: {analyzer.data[outcome].nunique()}")
            print(f"       - Value range: [{analyzer.data[outcome].min():.3f}, {analyzer.data[outcome].max():.3f}]")
            print(f"       - Mean: {analyzer.data[outcome].mean():.3f}")
            print(f"       - Std: {analyzer.data[outcome].std():.3f}")
            print(f"       - Sample values: {list(analyzer.data[outcome].head())}")
        
        # Check correlation
        if treatment in analyzer.data.columns and outcome in analyzer.data.columns:
            correlation = analyzer.data[treatment].corr(analyzer.data[outcome])
            print(f"DEBUG: Correlation between {treatment} and {outcome}: {correlation:.4f}")
          # Determine which data to use - CRITICAL: must match the graph columns
        use_encoded_data = False
        data_to_use = analyzer.data
        
        # Check if we're using encoded columns for the graph
        graph_uses_encoded = False
        if hasattr(analyzer.discovery, 'encoded_columns') and analyzer.discovery.encoded_columns:
            graph_uses_encoded = True
        
        # If the graph uses encoded variables OR we have encoded variables in treatment/outcome/confounders
        if (graph_uses_encoded or 
            treatment.endswith('_Code') or outcome.endswith('_Code') or 
            (confounders and any(c.endswith('_Code') for c in confounders))):
            if hasattr(analyzer.discovery, 'encoded_data') and analyzer.discovery.encoded_data is not None:
                print(f"DEBUG: Using encoded data because graph uses encoded variables")
                data_to_use = analyzer.discovery.encoded_data
                use_encoded_data = True
            else:
                print(f"DEBUG: WARNING: Graph uses encoded variables but no encoded data available!")
                print(f"DEBUG: Will attempt to use original data, but this may cause errors")
        else:
            print(f"DEBUG: Using original data for inference")
        
        # Verify variables exist in the data we're using
        missing_vars = []
        if treatment not in data_to_use.columns:
            missing_vars.append(f"treatment '{treatment}'")
        if outcome not in data_to_use.columns:
            missing_vars.append(f"outcome '{outcome}'")
        if confounders:
            for conf in confounders:
                if conf not in data_to_use.columns:
                    missing_vars.append(f"confounder '{conf}'")
        
        if missing_vars:
            error_msg = f"Variables not found in {'encoded' if use_encoded_data else 'original'} data: {', '.join(missing_vars)}"
            print(f"DEBUG: ERROR: {error_msg}")
            print(f"DEBUG: Available columns: {list(data_to_use.columns)}")
            raise ValueError(error_msg)
        
        print(f"DEBUG: Using {'encoded' if use_encoded_data else 'original'} data for inference")
        print(f"DEBUG: Data shape: {data_to_use.shape}")
        
        # Update variable stats with the data we're actually using
        print(f"DEBUG: Treatment '{treatment}' stats in data_to_use:")
        print(f"       - Unique values: {data_to_use[treatment].nunique()}")
        print(f"       - Value range: [{data_to_use[treatment].min():.3f}, {data_to_use[treatment].max():.3f}]")
        print(f"       - Mean: {data_to_use[treatment].mean():.3f}")
        print(f"       - Std: {data_to_use[treatment].std():.3f}")
        
        print(f"DEBUG: Outcome '{outcome}' stats in data_to_use:")
        print(f"       - Unique values: {data_to_use[outcome].nunique()}")
        print(f"       - Value range: [{data_to_use[outcome].min():.3f}, {data_to_use[outcome].max():.3f}]")
        print(f"       - Mean: {data_to_use[outcome].mean():.3f}")
        print(f"       - Std: {data_to_use[outcome].std():.3f}")
        
        correlation = data_to_use[treatment].corr(data_to_use[outcome])
        print(f"DEBUG: Correlation between {treatment} and {outcome} in data_to_use: {correlation:.4f}")
        
        print(f"DEBUG: ========================================================")
        
        # Only create DOT graph if we have an adjacency matrix from causal discovery
        graph = None
        if hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None:
            print("DEBUG: Using adjacency_matrix to build graph for DoWhy")
            
            # CRITICAL FIX: Always use the same columns that were used for causal discovery
            # The adjacency matrix dimensions must match exactly with the graph columns
            if hasattr(analyzer.discovery, 'encoded_columns') and analyzer.discovery.encoded_columns:
                graph_columns = analyzer.discovery.encoded_columns
                print(f"DEBUG: Using encoded columns for graph (matches adjacency matrix): {graph_columns}")
                print(f"DEBUG: Adjacency matrix shape: {analyzer.adjacency_matrix.shape}")
                print(f"DEBUG: Graph columns count: {len(graph_columns)}")
                
                # Verify the dimensions match
                if analyzer.adjacency_matrix.shape[0] != len(graph_columns):
                    print(f"DEBUG: CRITICAL ERROR: Adjacency matrix shape {analyzer.adjacency_matrix.shape} doesn't match encoded columns {len(graph_columns)}")
                    print(f"DEBUG: This will cause DoWhy to fail. Using fallback.")
                    graph_columns = None
            else:
                print(f"DEBUG: No encoded_columns available in discovery module")
                graph_columns = None
            
            if graph_columns is not None:
                dot_graph = _adjacency_to_dowhy_dot_graph(
                    analyzer.adjacency_matrix, 
                    graph_columns, 
                    treatment, 
                    outcome, 
                    confounders
                )
                graph = dot_graph
            else:
                print("DEBUG: Cannot create graph - no matching columns found")
        else:
            print("DEBUG: No adjacency_matrix available, not passing graph to DoWhy")
        
        # Always use confounders if provided
        if confounders:
            print(f"DEBUG: Using user-specified confounders: {confounders}")
        else:
            print("DEBUG: No confounders provided, relying on graph or DoWhy automatic confounder selection")        # Create causal model - only pass graph if we have one
        causal_model_args = {
            'data': data_to_use,  # Use the appropriate dataset
            'treatment': treatment,
            'outcome': outcome,
            'common_causes': confounders if confounders else None
        }
        
        if graph is not None:
            causal_model_args['graph'] = graph
            print("DEBUG: Passing graph to DoWhy causal model")
        else:
            print("DEBUG: Not passing graph to DoWhy causal model")
        
        print(f"DEBUG: Creating CausalModel with args:")
        print(f"       - data shape: {causal_model_args['data'].shape}")
        print(f"       - treatment: {causal_model_args['treatment']}")
        print(f"       - outcome: {causal_model_args['outcome']}")
        print(f"       - common_causes: {causal_model_args['common_causes']}")
        print(f"       - graph provided: {'graph' in causal_model_args}")
        
        causal_model = CausalModel(**causal_model_args)
        
        print(f"DEBUG: CausalModel created successfully")
        
        # Identify causal effect
        print(f"DEBUG: Identifying causal effect...")
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        print(f"DEBUG: Identified estimand: {identified_estimand}")
        
        # Estimate the effect using linear regression
        print(f"DEBUG: Estimating causal effect using linear regression...")
        causal_estimate = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            effect_modifiers=[]
        )
        
        print(f"DEBUG: Causal estimate value: {causal_estimate.value}")
        print(f"DEBUG: Causal estimate summary: {causal_estimate}")
        print(f"DEBUG: Causal estimate type: {type(causal_estimate)}")
        
        # Check if the estimate is None or essentially zero
        if causal_estimate.value is None:
            print(f"DEBUG: *** WARNING: Causal estimate is None - DoWhy couldn't find a valid estimand ***")
            # Return a zero effect with explanation
            return {
                "estimates": {
                    "Linear Regression": {
                        "estimate": 0.0,
                        "confidence_interval": [None, None],
                        "p_value": None,
                        "method": "backdoor.linear_regression",
                        "error": "No valid estimand found by DoWhy"
                    }
                },
                "consensus_estimate": 0.0,
                "robustness": {"status": "failed", "reason": "no_valid_estimand"},
                "interpretation": "No causal effect could be estimated due to DoWhy failing to identify a valid estimand",
                "recommendation": "Consider using different confounders or a different causal discovery method",
                "additional_metrics": {}
            }
        elif abs(causal_estimate.value) < 1e-10:
            print(f"DEBUG: *** WARNING: Estimate is extremely small ({causal_estimate.value}), likely indicating no effect or numerical issues ***")
        
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
            p_value = None        # Package results
        results = {
            "Linear Regression": {
                "estimate": causal_estimate.value,
                "confidence_interval": confidence_interval,
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
            "additional_metrics": calculate_simple_metrics(data_to_use, treatment, outcome, causal_estimate.value)
        }
