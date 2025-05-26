from typing import Dict

def analyze_heterogeneity(data, treatment: str, outcome: str, moderator: str = None) -> Dict:
    """Analyze heterogeneous treatment effects"""
    if not moderator:
        return {"error": "No moderator variable selected"}
    
    try:
        moderator_data = data[moderator]
        moderator_median = moderator_data.median()
        
        high_mod_data = data[moderator_data > moderator_median]
        high_mod_corr = high_mod_data[treatment].corr(high_mod_data[outcome])
        
        low_mod_data = data[moderator_data <= moderator_median]
        low_mod_corr = low_mod_data[treatment].corr(low_mod_data[outcome])
        
        effect_difference = high_mod_corr - low_mod_corr
        
        return {
            "high_moderator_effect": high_mod_corr,
            "low_moderator_effect": low_mod_corr,
            "effect_difference": effect_difference,
            "interpretation": f"The effect varies by {moderator}: {abs(effect_difference):.3f} difference between high/low groups"
        }
        
    except Exception as e:
        return {"error": str(e)}

def simulate_intervention(data, treatment: str, outcome: str, intervention_size: float) -> Dict:
    """Simulate the effect of a policy intervention"""
    try:
        current_outcome_mean = data[outcome].mean()
        correlation = data[treatment].corr(data[outcome])
        outcome_std = data[outcome].std()
        treatment_std = data[treatment].std()
        
        predicted_change = correlation * (outcome_std / treatment_std) * intervention_size
        predicted_new_outcome = current_outcome_mean + predicted_change
        percent_change = (predicted_change / current_outcome_mean) * 100
        
        return {
            "current_baseline": current_outcome_mean,
            "intervention_size": intervention_size,
            "predicted_outcome_change": predicted_change,
            "predicted_new_outcome": predicted_new_outcome,
            "percent_change": percent_change,
            "interpretation": f"Increasing {treatment} by {intervention_size} units may change {outcome} by {predicted_change:.3f} units ({percent_change:.1f}%)"
        }
        
    except Exception as e:
        return {"error": str(e)}
