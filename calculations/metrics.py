from typing import Dict

def classify_effect_size(effect: float) -> str:
    """Classify effect size"""
    abs_effect = abs(effect)
    if abs_effect < 0.01:
        return "Very Small"
    elif abs_effect < 0.1:
        return "Small"
    elif abs_effect < 0.5:
        return "Medium"
    else:
        return "Large"

def calculate_simple_metrics(data, treatment: str, outcome: str, estimate: float) -> Dict:
    """Calculate simplified metrics for speed"""
    additional_metrics = {}
    
    try:
        correlation = data[treatment].corr(data[outcome])
        r_squared = correlation ** 2
        additional_metrics["r_squared"] = r_squared
        additional_metrics["explained_variance_percent"] = r_squared * 100
        additional_metrics["effect_size_interpretation"] = classify_effect_size(estimate)
    except Exception as e:
        additional_metrics["error"] = str(e)
    
    return additional_metrics

def generate_simple_recommendation(estimates: Dict) -> str:
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

def interpret_ate(ate_value: float, treatment: str, outcome: str) -> str:
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
