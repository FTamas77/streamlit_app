import pandas as pd
import numpy as np
import warnings
from typing import Dict, List
import streamlit as st

# Try to import optional dependencies
try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

def calculate_ate_dowhy(analyzer, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
    """Calculate ATE using DoWhy (when available)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        from utils.constraint_utils import adjacency_to_graph_string
        graph_string = adjacency_to_graph_string(analyzer.adjacency_matrix, analyzer.data.columns)
        
        if not graph_string:
            edges = []
            if confounders:
                for conf in confounders:
                    edges.append(f'"{conf}" -> "{treatment}"')
                    edges.append(f'"{conf}" -> "{outcome}"')
            edges.append(f'"{treatment}" -> "{outcome}"')
            graph_string = "; ".join(edges)
        
        causal_model = CausalModel(
            data=analyzer.data,
            treatment=treatment,
            outcome=outcome,
            graph=graph_string,
            common_causes=confounders or []
        )
        
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        causal_estimate = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        
        try:
            confidence_interval = causal_estimate.get_confidence_intervals()
            p_value = causal_estimate.get_significance_test_results()['p_value']
        except:
            confidence_interval = [None, None]
            p_value = None
        
        results = {
            "Linear Regression": {
                "estimate": causal_estimate.value,
                "confidence_interval": confidence_interval,
                "p_value": p_value,
                "method": "backdoor.linear_regression"
            }
        }
        
        from calculations.metrics import calculate_simple_metrics, generate_simple_recommendation, interpret_ate
        
        return {
            "estimates": results,
            "consensus_estimate": causal_estimate.value,
            "robustness": {"status": "skipped_for_speed"},
            "interpretation": interpret_ate(causal_estimate.value, treatment, outcome),
            "recommendation": generate_simple_recommendation(results),
            "additional_metrics": calculate_simple_metrics(analyzer.data, treatment, outcome, causal_estimate.value)
        }

def calculate_ate_fallback(analyzer, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
    """Fallback ATE calculation using regression without DoWhy"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import scipy.stats as stats
        
        X = analyzer.data[[treatment] + (confounders or [])].values
        y = analyzer.data[outcome].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        treatment_effect = model.coef_[0]
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_treatment = np.sqrt(cov_matrix[1, 1])
        
        t_stat = treatment_effect / se_treatment
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - X.shape[1] - 1))
        
        t_critical = stats.t.ppf(0.975, len(X) - X.shape[1] - 1)
        ci_lower = treatment_effect - t_critical * se_treatment
        ci_upper = treatment_effect + t_critical * se_treatment
        
        results = {
            "Linear Regression (Fallback)": {
                "estimate": treatment_effect,
                "confidence_interval": [ci_lower, ci_upper],
                "p_value": p_value,
                "method": "sklearn_regression"
            }
        }
        
        from calculations.metrics import calculate_simple_metrics, generate_simple_recommendation, interpret_ate
        
        return {
            "estimates": results,
            "consensus_estimate": treatment_effect,
            "robustness": {"status": "fallback_method"},
            "interpretation": interpret_ate(treatment_effect, treatment, outcome),
            "recommendation": generate_simple_recommendation(results),
            "additional_metrics": calculate_simple_metrics(analyzer.data, treatment, outcome, treatment_effect)
        }
        
    except Exception as e:
        # Final fallback to correlation
        correlation = analyzer.data[treatment].corr(analyzer.data[outcome])
        from calculations.metrics import classify_effect_size
        
        return {
            "estimates": {"Simple Correlation": {"estimate": correlation, "confidence_interval": [None, None], "p_value": None}},
            "consensus_estimate": correlation,
            "robustness": {"warning": "Only correlation available"},
            "interpretation": f"Simple correlation shows {abs(correlation):.4f} {'positive' if correlation > 0 else 'negative'} association",
            "recommendation": "⚠️ Use caution: This is correlation, not causation.",
            "additional_metrics": {"r_squared": correlation**2, "effect_size_interpretation": classify_effect_size(correlation)}
        }
