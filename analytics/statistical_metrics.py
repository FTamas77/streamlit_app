from typing import Dict

def calculate_simple_metrics(data, treatment: str, outcome: str, estimate: float) -> Dict:
    """Calculate simplified statistical metrics for general analysis
    
    Note: These are correlation-based metrics, not causal metrics.
    Use only for general statistical analysis, not causal inference.
    """
    additional_metrics = {}
    
    try:
        correlation = data[treatment].corr(data[outcome])
        r_squared = correlation ** 2
        additional_metrics["r_squared"] = r_squared
        additional_metrics["explained_variance_percent"] = r_squared * 100        # Import from utils.effect_size for effect size classification
        from utils.effect_size import classify_effect_size
        additional_metrics["effect_size_interpretation"] = classify_effect_size(estimate)
    except Exception as e:
        additional_metrics["error"] = str(e)
    
    return additional_metrics
