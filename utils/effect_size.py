def classify_effect_size(effect: float) -> str:
    """Classify effect size - general statistical utility
    
    This is a general utility for classifying effect sizes that can be used
    by both causal analysis and general statistical analysis.
    """
    abs_effect = abs(effect)
    if abs_effect < 0.01:
        return "Very Small"
    elif abs_effect < 0.1:
        return "Small"
    elif abs_effect < 0.5:
        return "Medium"
    else:
        return "Large"
