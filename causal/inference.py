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
    
    # Input validation
    if not analyzer:
        raise ValueError("Analyzer object is required but not provided")
    
    if not treatment or not isinstance(treatment, str):
        raise ValueError(f"Treatment must be a non-empty string, got: {treatment}")
    
    if not outcome or not isinstance(outcome, str):
        raise ValueError(f"Outcome must be a non-empty string, got: {outcome}")
    
    if treatment == outcome:
        raise ValueError("Treatment and outcome variables cannot be the same")
    
    if confounders is not None:
        if not isinstance(confounders, list):
            raise ValueError("Confounders must be a list or None")
        if treatment in confounders:
            raise ValueError("Treatment variable cannot be included as a confounder")
        if outcome in confounders:
            raise ValueError("Outcome variable cannot be included as a confounder")
        # Remove duplicates while preserving order
        confounders = list(dict.fromkeys(confounders))
    
    # Check if analyzer has required data
    if not hasattr(analyzer, 'data') or analyzer.data is None:
        raise ValueError("Analyzer must have data loaded")
    
    if analyzer.data.empty:
        raise ValueError("Analyzer data is empty")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            from analytics.statistical_metrics import calculate_simple_metrics
            from utils.effect_size import classify_effect_size
        except ImportError as e:
            print(f"DEBUG: Warning - Could not import metrics modules: {e}")
            # Create fallback functions
            def calculate_simple_metrics(data, treatment, outcome, effect):
                return {}
            def classify_effect_size(effect):
                return "unknown"
        
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
            try:
                print(f"DEBUG: encoded_data available with shape: {analyzer.discovery.encoded_data.shape}")
                print(f"DEBUG: encoded_data columns: {list(analyzer.discovery.encoded_data.columns)}")
                print(f"DEBUG: Treatment '{treatment}' in encoded_data: {treatment in analyzer.discovery.encoded_data.columns}")
                print(f"DEBUG: Outcome '{outcome}' in encoded_data: {outcome in analyzer.discovery.encoded_data.columns}")
            except (TypeError, AttributeError):
                # Handle Mock objects in tests
                print(f"DEBUG: encoded_data available (Mock object in tests)")
        else:
            print(f"DEBUG: No encoded_data available")
          # Check variable values and distributions
        if treatment in analyzer.data.columns:
            print(f"DEBUG: Treatment '{treatment}' stats in analyzer.data:")
            print(f"       - Unique values: {analyzer.data[treatment].nunique()}")
            # Check if treatment is numeric for proper formatting
            if analyzer.data[treatment].dtype.kind in 'biufc':  # numeric types
                print(f"       - Value range: [{analyzer.data[treatment].min():.3f}, {analyzer.data[treatment].max():.3f}]")
                print(f"       - Mean: {analyzer.data[treatment].mean():.3f}")
                print(f"       - Std: {analyzer.data[treatment].std():.3f}")
            else:
                print(f"       - Value range: [{analyzer.data[treatment].min()}, {analyzer.data[treatment].max()}]")
                print(f"       - Type: categorical")
            print(f"       - Sample values: {list(analyzer.data[treatment].head())}")
        
        if outcome in analyzer.data.columns:
            print(f"DEBUG: Outcome '{outcome}' stats in analyzer.data:")
            print(f"       - Unique values: {analyzer.data[outcome].nunique()}")
            # Check if outcome is numeric for proper formatting
            if analyzer.data[outcome].dtype.kind in 'biufc':  # numeric types
                print(f"       - Value range: [{analyzer.data[outcome].min():.3f}, {analyzer.data[outcome].max():.3f}]")
                print(f"       - Mean: {analyzer.data[outcome].mean():.3f}")
                print(f"       - Std: {analyzer.data[outcome].std():.3f}")
            else:
                print(f"       - Value range: [{analyzer.data[outcome].min()}, {analyzer.data[outcome].max()}]")
                print(f"       - Type: categorical")
            print(f"       - Sample values: {list(analyzer.data[outcome].head())}")
          # Check correlation
        if treatment in analyzer.data.columns and outcome in analyzer.data.columns:
            # Calculate correlation if both variables are numeric
            if (analyzer.data[treatment].dtype.kind in 'biufc' and 
                analyzer.data[outcome].dtype.kind in 'biufc'):
                correlation = analyzer.data[treatment].corr(analyzer.data[outcome])
                print(f"DEBUG: Correlation between {treatment} and {outcome}: {correlation:.4f}")
            else:
                print(f"DEBUG: Correlation not calculated (at least one variable is categorical)")
        
        # Determine which data to use - CRITICAL: must match the graph columns
        use_encoded_data = False
        data_to_use = analyzer.data
          # Check if we're using encoded columns for the graph
        graph_uses_encoded = False
        if hasattr(analyzer.discovery, 'columns') and analyzer.discovery.columns:
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
          # Enhanced variable validation with detailed error messages
        missing_vars = []
        invalid_vars = []
        
        # Handle Mock objects in tests
        try:
            data_columns = list(data_to_use.columns)
        except (TypeError, AttributeError):
            # Mock object in tests - use original data columns
            data_columns = list(analyzer.data.columns)
            data_to_use = analyzer.data
        
        # Check treatment variable
        if treatment not in data_columns:
            missing_vars.append(f"treatment '{treatment}'")
        else:
            # Validate treatment variable characteristics
            treatment_data = data_to_use[treatment]
            if treatment_data.isnull().all():
                invalid_vars.append(f"treatment '{treatment}' contains only null values")
            elif treatment_data.nunique() < 2:
                invalid_vars.append(f"treatment '{treatment}' has insufficient variation (only {treatment_data.nunique()} unique value)")
            elif not np.issubdtype(treatment_data.dtype, np.number):
                invalid_vars.append(f"treatment '{treatment}' is not numeric (type: {treatment_data.dtype})")
          # Check outcome variable
        if outcome not in data_columns:
            missing_vars.append(f"outcome '{outcome}'")
        else:
            # Validate outcome variable characteristics
            outcome_data = data_to_use[outcome]
            if outcome_data.isnull().all():
                invalid_vars.append(f"outcome '{outcome}' contains only null values")
            elif outcome_data.nunique() < 2:
                invalid_vars.append(f"outcome '{outcome}' has insufficient variation (only {outcome_data.nunique()} unique values)")
            elif not np.issubdtype(outcome_data.dtype, np.number):
                invalid_vars.append(f"outcome '{outcome}' is not numeric (type: {outcome_data.dtype})")
          # Check confounder variables
        if confounders:
            for conf in confounders:
                if conf not in data_columns:
                    missing_vars.append(f"confounder '{conf}'")
                else:
                    # Validate confounder characteristics
                    conf_data = data_to_use[conf]
                    if conf_data.isnull().all():
                        invalid_vars.append(f"confounder '{conf}' contains only null values")
                    elif not np.issubdtype(conf_data.dtype, np.number):
                        invalid_vars.append(f"confounder '{conf}' is not numeric (type: {conf_data.dtype})")
        
        # Report validation errors
        if missing_vars:
            error_msg = f"Variables not found in {'encoded' if use_encoded_data else 'original'} data: {', '.join(missing_vars)}"
            print(f"DEBUG: ERROR: {error_msg}")
            print(f"DEBUG: Available columns: {data_columns}")
            raise ValueError(error_msg)
        
        if invalid_vars:
            error_msg = f"Invalid variable characteristics: {'; '.join(invalid_vars)}"
            print(f"DEBUG: ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check for sufficient data
        non_null_rows = data_to_use[[treatment, outcome] + (confounders or [])].dropna()
        if len(non_null_rows) < 10:
            error_msg = f"Insufficient data after removing null values: only {len(non_null_rows)} rows available (minimum 10 required)"
            print(f"DEBUG: ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Update data_to_use to clean data
        if len(non_null_rows) < len(data_to_use):
            print(f"DEBUG: Removing {len(data_to_use) - len(non_null_rows)} rows with null values")
            data_to_use = non_null_rows
        
        print(f"DEBUG: Using {'encoded' if use_encoded_data else 'original'} data for inference")
        print(f"DEBUG: Data shape: {data_to_use.shape}")
          # Update variable stats with the data we're actually using
        print(f"DEBUG: Treatment '{treatment}' stats in data_to_use:")
        print(f"       - Unique values: {data_to_use[treatment].nunique()}")
        # Check if treatment is numeric for proper formatting
        if data_to_use[treatment].dtype.kind in 'biufc':  # numeric types
            print(f"       - Value range: [{data_to_use[treatment].min():.3f}, {data_to_use[treatment].max():.3f}]")
            print(f"       - Mean: {data_to_use[treatment].mean():.3f}")
            print(f"       - Std: {data_to_use[treatment].std():.3f}")
        else:
            print(f"       - Value range: [{data_to_use[treatment].min()}, {data_to_use[treatment].max()}]")
            print(f"       - Type: categorical")
        
        print(f"DEBUG: Outcome '{outcome}' stats in data_to_use:")
        print(f"       - Unique values: {data_to_use[outcome].nunique()}")
        # Check if outcome is numeric for proper formatting
        if data_to_use[outcome].dtype.kind in 'biufc':  # numeric types
            print(f"       - Value range: [{data_to_use[outcome].min():.3f}, {data_to_use[outcome].max():.3f}]")
            print(f"       - Mean: {data_to_use[outcome].mean():.3f}")
            print(f"       - Std: {data_to_use[outcome].std():.3f}")
        else:
            print(f"       - Value range: [{data_to_use[outcome].min()}, {data_to_use[outcome].max()}]")
            print(f"       - Type: categorical")
          # Calculate correlation if both variables are numeric
        if (data_to_use[treatment].dtype.kind in 'biufc' and 
            data_to_use[outcome].dtype.kind in 'biufc'):
            correlation = data_to_use[treatment].corr(data_to_use[outcome])
            print(f"DEBUG: Correlation between {treatment} and {outcome} in data_to_use: {correlation:.4f}")
        else:
            print(f"DEBUG: Correlation not calculated (at least one variable is categorical)")
        
        print(f"DEBUG: ========================================================")
        
        # Only create DOT graph if we have an adjacency matrix from causal discovery
        graph = None
        if hasattr(analyzer, 'adjacency_matrix') and analyzer.adjacency_matrix is not None:
            print("DEBUG: Using adjacency_matrix to build graph for DoWhy")            # CRITICAL FIX: Always use the same columns that were used for causal discovery
            # The adjacency matrix dimensions must match exactly with the graph columns
            if hasattr(analyzer.discovery, 'columns') and analyzer.discovery.columns:
                graph_columns = analyzer.discovery.columns
                print(f"DEBUG: Using columns for graph (matches adjacency matrix): {graph_columns}")
                print(f"DEBUG: Adjacency matrix shape: {analyzer.adjacency_matrix.shape}")
                
                # Handle Mock objects in tests
                try:
                    graph_columns_count = len(graph_columns)
                    print(f"DEBUG: Graph columns count: {graph_columns_count}")
                    
                    # Verify the dimensions match
                    if analyzer.adjacency_matrix.shape[0] != graph_columns_count:
                        print(f"DEBUG: CRITICAL ERROR: Adjacency matrix shape {analyzer.adjacency_matrix.shape} doesn't match encoded columns {graph_columns_count}")
                        print(f"DEBUG: This will cause DoWhy to fail. Using fallback.")
                        graph_columns = None
                except (TypeError, AttributeError):
                    # Mock object in tests - use fallback to original data columns
                    print(f"DEBUG: Mock columns in tests - using original data columns")
                    graph_columns = list(analyzer.data.columns)
            else:
                print(f"DEBUG: No columns available in discovery module")
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
        
        try:
            causal_model = CausalModel(**causal_model_args)
            print(f"DEBUG: CausalModel created successfully")
        except Exception as e:
            error_msg = f"Failed to create DoWhy CausalModel: {str(e)}"
            print(f"DEBUG: ERROR: {error_msg}")
            print(f"DEBUG: This might be due to graph/data inconsistencies or invalid variable specifications")
            raise RuntimeError(error_msg) from e
        
        # Identify causal effect with error handling
        print(f"DEBUG: Identifying causal effect...")
        try:
            identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
            print(f"DEBUG: Identified estimand: {identified_estimand}")
            
            # Check if DoWhy found any valid estimands
            if hasattr(identified_estimand, 'estimands') and len(identified_estimand.estimands) == 0:
                print(f"DEBUG: WARNING: No estimands identified by DoWhy")
            elif str(identified_estimand).strip() == "":
                print(f"DEBUG: WARNING: Empty estimand returned by DoWhy")
                
        except Exception as e:
            error_msg = f"Failed to identify causal effect: {str(e)}"
            print(f"DEBUG: ERROR: {error_msg}")
            # Return a structured error response instead of raising
            return {
                "estimates": {
                    "Linear Regression": {
                        "estimate": 0.0,
                        "confidence_interval": [None, None],
                        "p_value": None,
                        "method": "backdoor.linear_regression",
                        "error": f"Identification failed: {str(e)}"
                    }
                },
                "consensus_estimate": 0.0,
                "robustness": {"status": "failed", "reason": "identification_error"},
                "interpretation": f"Causal effect identification failed: {str(e)}",
                "recommendation": "Check data quality, variable specifications, or try different confounders",
                "additional_metrics": {}
            }
        
        # Estimate the effect using linear regression with error handling
        print(f"DEBUG: Estimating causal effect using linear regression...")
        try:
            causal_estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                effect_modifiers=[]
            )
            
            print(f"DEBUG: Causal estimate value: {causal_estimate.value}")
            print(f"DEBUG: Causal estimate summary: {causal_estimate}")
            print(f"DEBUG: Causal estimate type: {type(causal_estimate)}")
            
        except Exception as e:
            error_msg = f"Failed to estimate causal effect: {str(e)}"
            print(f"DEBUG: ERROR: {error_msg}")
            # Try alternative estimation methods
            alternative_methods = [
                "backdoor.propensity_score_matching",
                "backdoor.propensity_score_stratification",
                "backdoor.generalized_linear_model"
            ]
            
            causal_estimate = None
            for alt_method in alternative_methods:
                try:
                    print(f"DEBUG: Trying alternative method: {alt_method}")
                    causal_estimate = causal_model.estimate_effect(
                        identified_estimand,
                        method_name=alt_method
                    )
                    print(f"DEBUG: Alternative method {alt_method} succeeded")
                    break
                except Exception as alt_e:
                    print(f"DEBUG: Alternative method {alt_method} failed: {str(alt_e)}")
                    continue
            
            if causal_estimate is None:
                # Return structured error response
                return {
                    "estimates": {
                        "Linear Regression": {
                            "estimate": 0.0,
                            "confidence_interval": [None, None],
                            "p_value": None,
                            "method": "backdoor.linear_regression",
                            "error": f"All estimation methods failed. Last error: {str(e)}"
                        }
                    },
                    "consensus_estimate": 0.0,
                    "robustness": {"status": "failed", "reason": "estimation_error"},
                    "interpretation": f"Causal effect estimation failed: {str(e)}",
                    "recommendation": "Check data quality, try different methods, or consider alternative causal discovery approaches",
                    "additional_metrics": {}
                }
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
        
        # Extract confidence intervals and p-value with enhanced error handling
        confidence_interval = [None, None]
        p_value = None
        
        try:
            ci_result = causal_estimate.get_confidence_intervals()
            print(f"DEBUG: Raw confidence interval result: {ci_result}")
              # Handle different confidence interval formats
            if ci_result is not None:
                if isinstance(ci_result, np.ndarray):
                    # Handle numpy array format
                    if ci_result.ndim == 2 and ci_result.shape[1] == 2:
                        # Format: [[lower, upper]]
                        confidence_interval = [float(ci_result[0][0]), float(ci_result[0][1])]
                    elif ci_result.ndim == 1 and len(ci_result) == 2:
                        # Format: [lower, upper]
                        confidence_interval = [float(ci_result[0]), float(ci_result[1])]
                    else:
                        print(f"DEBUG: Unexpected numpy array shape: {ci_result.shape}")
                elif isinstance(ci_result, (list, tuple)) and len(ci_result) > 0:
                    if isinstance(ci_result[0], (list, tuple)) and len(ci_result[0]) == 2:
                        # Format: [[lower, upper]]
                        confidence_interval = [float(ci_result[0][0]), float(ci_result[0][1])]
                    elif len(ci_result) == 2 and isinstance(ci_result[0], (int, float)):
                        # Format: [lower, upper]
                        confidence_interval = [float(ci_result[0]), float(ci_result[1])]
                    else:                        print(f"DEBUG: Unexpected confidence interval format: {ci_result}")
                elif isinstance(ci_result, dict):
                    # Handle dictionary format
                    if 'lower' in ci_result and 'upper' in ci_result:
                        confidence_interval = [float(ci_result['lower']), float(ci_result['upper'])]
                    else:
                        print(f"DEBUG: Unexpected confidence interval dict format: {ci_result}")
                else:
                    print(f"DEBUG: Unexpected confidence interval type: {type(ci_result)}")
            
            print(f"DEBUG: Processed confidence interval: {confidence_interval}")
            
        except Exception as e:
            print(f"DEBUG: Failed to extract confidence intervals: {str(e)}")
            confidence_interval = [None, None]
        
        try:
            # Get p-value using test_stat_significance
            sig_results = causal_estimate.test_stat_significance()
            print(f"DEBUG: Significance test results: {sig_results}")
            
            if isinstance(sig_results, dict):
                if 'p_value' in sig_results:
                    p_val = sig_results['p_value']
                    # Handle numpy arrays
                    if isinstance(p_val, np.ndarray):
                        if p_val.size == 1:
                            p_value = float(p_val.item())
                        else:
                            p_value = float(p_val[0])
                    else:
                        p_value = float(p_val)
                elif 'p' in sig_results:
                    p_val = sig_results['p']
                    if isinstance(p_val, np.ndarray):
                        if p_val.size == 1:
                            p_value = float(p_val.item())
                        else:
                            p_value = float(p_val[0])
                    else:
                        p_value = float(p_val)
                else:
                    print(f"DEBUG: No p_value found in significance results: {list(sig_results.keys())}")
            else:
                print(f"DEBUG: Unexpected significance test result type: {type(sig_results)}")
                
            print(f"DEBUG: Processed p-value: {p_value}")
            
        except Exception as e:
            print(f"DEBUG: Failed to extract p-value: {str(e)}")
            p_value = None
        
        # Validate the final estimate
        final_estimate = causal_estimate.value
        if not isinstance(final_estimate, (int, float)):
            print(f"DEBUG: WARNING: Estimate is not numeric: {type(final_estimate)}, value: {final_estimate}")
            try:
                final_estimate = float(final_estimate)
            except (ValueError, TypeError):
                print(f"DEBUG: Could not convert estimate to float, using 0.0")
                final_estimate = 0.0
        
        # Check for invalid values
        if np.isnan(final_estimate) or np.isinf(final_estimate):
            print(f"DEBUG: WARNING: Estimate is NaN or Inf: {final_estimate}, using 0.0")
            final_estimate = 0.0        # Package results with additional validation
        try:
            method_name = "backdoor.linear_regression"
            if hasattr(causal_estimate, 'method_name'):
                method_name = causal_estimate.method_name
            
            results = {
                "Linear Regression": {
                    "estimate": final_estimate,
                    "confidence_interval": confidence_interval,
                    "p_value": p_value,
                    "method": method_name
                }
            }
            
            # Calculate additional metrics with error handling
            additional_metrics = {}
            try:
                additional_metrics = calculate_simple_metrics(data_to_use, treatment, outcome, final_estimate)
            except Exception as e:
                print(f"DEBUG: Failed to calculate additional metrics: {str(e)}")
                additional_metrics = {"error": str(e)}
            
            # Generate interpretation and recommendation
            try:
                interpretation = _interpret_ate(final_estimate, treatment, outcome)
                recommendation = _generate_simple_recommendation(results)
            except Exception as e:
                print(f"DEBUG: Failed to generate interpretation/recommendation: {str(e)}")
                interpretation = f"Causal effect estimate: {final_estimate}"
                recommendation = "Unable to generate recommendation due to processing error"
            
            # Determine robustness status
            robustness_status = "completed"
            if abs(final_estimate) < 1e-10:
                robustness_status = "low_effect_detected"
            elif p_value is not None and p_value > 0.05:
                robustness_status = "not_significant"
            
            final_result = {
                "estimates": results,
                "consensus_estimate": final_estimate,
                "robustness": {"status": robustness_status},
                "interpretation": interpretation,
                "recommendation": recommendation,
                "additional_metrics": additional_metrics
            }
            
            print(f"DEBUG: Final result prepared successfully")
            print(f"DEBUG: Consensus estimate: {final_result['consensus_estimate']}")
            print(f"DEBUG: ================ END CAUSAL INFERENCE DEBUG ================")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Failed to package final results: {str(e)}"
            print(f"DEBUG: ERROR: {error_msg}")
            # Return basic error response
            return {
                "estimates": {
                    "Linear Regression": {
                        "estimate": 0.0,
                        "confidence_interval": [None, None],
                        "p_value": None,
                        "method": "backdoor.linear_regression",
                        "error": error_msg
                    }
                },
                "consensus_estimate": 0.0,
                "robustness": {"status": "failed", "reason": "packaging_error"},
                "interpretation": f"Result packaging failed: {str(e)}",
                "recommendation": "Contact support - internal processing error",
                "additional_metrics": {}
            }
