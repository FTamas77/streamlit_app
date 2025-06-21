import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings

def run_traditional_analysis(analyzer, treatment, outcome):
    """
    Run traditional statistical analysis for comparison with causal inference.
    
    Args:
        analyzer: CausalAnalyzer instance with data
        treatment: Treatment variable name
        outcome: Outcome variable name
        
    Returns:
        Dict with traditional analysis results
    """
    data = analyzer.data.copy()
    results = {
        'methods': {},
        'warnings': [],
        'assumptions': []
    }
    
    # Handle encoded data if available
    if hasattr(analyzer.discovery, 'encoded_data') and analyzer.discovery.encoded_data is not None:
        # Use encoded data for consistency with causal analysis
        encoded_data = analyzer.discovery.encoded_data.copy()
        treatment_data = encoded_data[treatment]
        outcome_data = encoded_data[outcome]
        data_for_analysis = encoded_data
    else:
        # Encode categorical variables for traditional analysis
        data_for_analysis = data.copy()
        le_treatment = LabelEncoder()
        le_outcome = LabelEncoder()
        
        if data[treatment].dtype == 'object':
            treatment_data = le_treatment.fit_transform(data[treatment])
            data_for_analysis[treatment] = treatment_data
            results['warnings'].append(f"Treatment '{treatment}' was label-encoded for traditional analysis")
        else:
            treatment_data = data[treatment]
            
        if data[outcome].dtype == 'object':
            outcome_data = le_outcome.fit_transform(data[outcome])
            data_for_analysis[outcome] = outcome_data
            results['warnings'].append(f"Outcome '{outcome}' was label-encoded for traditional analysis")
        else:
            outcome_data = data[outcome]
    
    # 1. Simple Correlation Analysis
    try:
        correlation = np.corrcoef(treatment_data, outcome_data)[0, 1]
        results['methods']['correlation'] = {
            'name': 'Simple Correlation',
            'estimate': correlation,
            'description': 'Pearson correlation coefficient between treatment and outcome',
            'assumptions': ['Linear relationship', 'No confounders considered'],
            'interpretation': f"{'Positive' if correlation > 0 else 'Negative'} correlation of {abs(correlation):.4f}"
        }
    except Exception as e:
        results['methods']['correlation'] = {
            'name': 'Simple Correlation',
            'error': str(e)
        }
    
    # 2. Simple Linear Regression (Treatment → Outcome, no confounders)
    try:
        X_simple = treatment_data.values.reshape(-1, 1)
        y = outcome_data.values
        
        reg_simple = LinearRegression().fit(X_simple, y)
        effect_simple = reg_simple.coef_[0]
        r2_simple = reg_simple.score(X_simple, y)
        
        results['methods']['simple_regression'] = {
            'name': 'Simple Linear Regression',
            'estimate': effect_simple,
            'r_squared': r2_simple,
            'description': 'Linear regression: Outcome ~ Treatment (no confounders)',
            'assumptions': ['Linear relationship', 'Independence', 'Homoscedasticity', 'No confounders'],
            'interpretation': f"Each unit increase in treatment changes outcome by {effect_simple:.4f}"
        }
    except Exception as e:
        results['methods']['simple_regression'] = {
            'name': 'Simple Linear Regression',
            'error': str(e)
        }
      # 3. Multiple Linear Regression (with available confounders from causal discovery)
    # If causal discovery was run, use confounders identified from the graph
    confounders = []
    if hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None:
        if hasattr(analyzer.discovery, 'columns') and analyzer.discovery.columns:
            # Get all variables except treatment and outcome as potential confounders
            all_vars = list(analyzer.discovery.columns)
            confounders = [var for var in all_vars if var not in [treatment, outcome]]
            if confounders:
                print(f"DEBUG: Using confounders from causal graph: {confounders}")
    
    if confounders and len(confounders) > 0:
        try:
            # Prepare features including confounders
            feature_cols = [treatment] + confounders
            X_multi = data_for_analysis[feature_cols].values
            y = outcome_data.values
            
            # Handle any remaining categorical variables
            for i, col in enumerate(feature_cols):
                if data_for_analysis[col].dtype == 'object':
                    le = LabelEncoder()
                    X_multi[:, i] = le.fit_transform(data_for_analysis[col])
            
            reg_multi = LinearRegression().fit(X_multi, y)
            effect_multi = reg_multi.coef_[0]  # Treatment coefficient
            r2_multi = reg_multi.score(X_multi, y)
            
            results['methods']['multiple_regression'] = {
                'name': 'Multiple Linear Regression',
                'estimate': effect_multi,
                'r_squared': r2_multi,
                'confounders_included': confounders,
                'description': f'Linear regression: Outcome ~ Treatment + {" + ".join(confounders)}',
                'assumptions': ['Linear relationship', 'Independence', 'Homoscedasticity', 'No unobserved confounders'],
                'interpretation': f"Treatment effect: {effect_multi:.4f} (controlling for {len(confounders)} confounders)"
            }
        except Exception as e:
            results['methods']['multiple_regression'] = {
                'name': 'Multiple Linear Regression',
                'error': str(e)
            }
    
    # 4. T-test (if treatment is binary)
    unique_treatments = len(np.unique(treatment_data))
    if unique_treatments == 2:
        try:
            group_0 = outcome_data[treatment_data == np.unique(treatment_data)[0]]
            group_1 = outcome_data[treatment_data == np.unique(treatment_data)[1]]
            
            t_stat, p_value = stats.ttest_ind(group_1, group_0)
            effect_ttest = np.mean(group_1) - np.mean(group_0)
            
            results['methods']['t_test'] = {
                'name': 'Independent T-Test',
                'estimate': effect_ttest,
                'p_value': p_value,
                't_statistic': t_stat,
                'description': 'Two-sample t-test comparing treatment groups',
                'assumptions': ['Normal distribution', 'Equal variances', 'Independence', 'No confounders'],
                'interpretation': f"Mean difference: {effect_ttest:.4f} (p={p_value:.4f})"
            }
        except Exception as e:
            results['methods']['t_test'] = {
                'name': 'Independent T-Test',
                'error': str(e)
            }
    
    # Add general assumptions and limitations
    results['assumptions'] = [
        "Traditional methods assume no unobserved confounding",
        "Linear relationships between variables",
        "Statistical independence of observations",
        "Proper model specification"
    ]
    
    results['limitations'] = [
        "Cannot distinguish correlation from causation",
        "Vulnerable to confounding bias",
        "May miss non-linear relationships",
        "Assumes treatment assignment is random (often violated)"
    ]
    
    return results

def compare_causal_vs_traditional(causal_results, traditional_results, treatment, outcome):
    """
    Create a comparison between causal and traditional analysis results.
    
    Args:
        causal_results: Results from causal inference
        traditional_results: Results from traditional analysis
        treatment: Treatment variable name
        outcome: Outcome variable name
        
    Returns:
        Dict with comparison insights
    """
    comparison = {
        'summary': {},
        'key_differences': [],
        'business_implications': []
    }
    
    causal_estimate = causal_results.get('consensus_estimate', 0)
    
    # Compare estimates
    traditional_estimates = {}
    for method_name, method_results in traditional_results['methods'].items():
        if 'estimate' in method_results and 'error' not in method_results:
            traditional_estimates[method_name] = method_results['estimate']
    
    comparison['summary'] = {
        'causal_estimate': causal_estimate,
        'traditional_estimates': traditional_estimates
    }
    
    # Identify key differences
    for method_name, trad_estimate in traditional_estimates.items():
        difference = abs(causal_estimate - trad_estimate)
        pct_difference = (difference / max(abs(causal_estimate), abs(trad_estimate), 0.0001)) * 100
        
        if pct_difference > 10:  # Significant difference
            comparison['key_differences'].append({
                'method': method_name,
                'causal': causal_estimate,
                'traditional': trad_estimate,
                'difference': difference,
                'percent_difference': pct_difference,
                'interpretation': f"Causal AI estimate differs by {pct_difference:.1f}% from {method_name}"
            })
    
    # Business implications
    comparison['business_implications'] = [
        "🎯 **Causal AI Advantage**: Accounts for confounding variables that traditional methods miss",
        "⚠️ **Traditional Risk**: May overestimate or underestimate true causal effects",
        "🔬 **Scientific Rigor**: Causal inference provides stronger evidence for decision-making",
        "💼 **ROI Impact**: More accurate effect estimates lead to better resource allocation"
    ]
    
    return comparison
