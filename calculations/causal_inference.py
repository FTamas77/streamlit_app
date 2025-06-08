import pandas as pd
import numpy as np
import warnings
from typing import Dict, List

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

def calculate_ate_dowhy(analyzer, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
    """Calculate ATE using DoWhy - prioritizes user-specified confounders over graph structure"""
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for ATE calculation but not available")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        from utils.constraint_utils import adjacency_to_dowhy_dot_graph
        from calculations.metrics import calculate_simple_metrics, generate_simple_recommendation, interpret_ate
        
        # Only create DOT graph if we have an adjacency matrix from causal discovery
        graph = None
        if hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None:
            dot_graph = adjacency_to_dowhy_dot_graph(
                analyzer.adjacency_matrix, 
                analyzer.data.columns.tolist(), 
                treatment, 
                outcome, 
                confounders
            )
            graph = dot_graph
        
        # Create causal model - only pass graph if we have one
        causal_model_args = {
            'data': analyzer.data,
            'treatment': treatment,
            'outcome': outcome,
            'common_causes': confounders if confounders else None
        }
        
        if graph is not None:
            causal_model_args['graph'] = graph
            
        causal_model = CausalModel(**causal_model_args)
        
        # Identify causal effect
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        
        # Log which approach is being used
        if confounders:
            print(f"Using user-specified confounders: {confounders}")
        else:
            print("Using graph-based automatic confounder identification")
        
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
            "interpretation": interpret_ate(causal_estimate.value, treatment, outcome),
            "recommendation": generate_simple_recommendation(results),
            "additional_metrics": calculate_simple_metrics(analyzer.data, treatment, outcome, causal_estimate.value)
        }
