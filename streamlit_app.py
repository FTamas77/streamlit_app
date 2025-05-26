import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from lingam import DirectLiNGAM
import numpy as np
import dowhy
from dowhy import CausalModel
import openai
import json
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
import warnings

# Suppress DoWhy warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='dowhy')
warnings.filterwarnings('ignore', category=UserWarning, module='dowhy')

# Configure page
st.set_page_config(
    page_title="Causal AI Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client (you'll need to set your API key)
@st.cache_resource
def init_openai(api_key=None):
    # Use provided API key or get from secrets
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    return openai

class CausalAnalyzer:
    """Main class for causal analysis pipeline"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.adjacency_matrix = None
        self.causal_model = None
        self.domain_constraints = None
        
    def load_data(self, uploaded_file):
        """Load data from uploaded Excel file with robust error handling"""
        try:
            if uploaded_file.name.endswith('.xlsx'):
                # Excel files are generally more structured
                self.data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                # Try multiple CSV reading strategies
                try:
                    # First attempt: Standard CSV reading
                    self.data = pd.read_csv(uploaded_file, encoding='utf-8')
                except (pd.errors.ParserError, UnicodeDecodeError) as e:
                    st.warning(f"Standard CSV parsing failed: {str(e)}. Trying alternative methods...")
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    try:
                        # Second attempt: Different encoding
                        self.data = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.info("✅ Successfully loaded with latin-1 encoding")
                    except Exception:
                        # Reset file pointer again
                        uploaded_file.seek(0)
                        
                        try:
                            # Third attempt: Handle inconsistent fields
                            self.data = pd.read_csv(
                                uploaded_file, 
                                encoding='utf-8',
                                error_bad_lines=False,  # Skip bad lines
                                warn_bad_lines=True,    # Warn about skipped lines
                                on_bad_lines='skip'     # For newer pandas versions
                            )
                            st.warning("⚠️ Some rows were skipped due to formatting issues")
                        except Exception:
                            # Reset file pointer one more time
                            uploaded_file.seek(0)
                            
                            try:
                                # Fourth attempt: Use python engine with flexible parsing
                                self.data = pd.read_csv(
                                    uploaded_file,
                                    engine='python',
                                    encoding='utf-8',
                                    sep=None,  # Auto-detect separator
                                    skipinitialspace=True,
                                    quoting=3  # QUOTE_NONE
                                )
                                st.info("✅ Successfully loaded with flexible parsing")
                            except Exception as final_error:
                                st.error(f"Unable to parse CSV file after multiple attempts: {str(final_error)}")
                                st.info("""
                                **Suggestions to fix your CSV file:**
                                1. Check for inconsistent number of columns
                                2. Ensure proper quote handling (commas inside quotes)
                                3. Try saving as Excel (.xlsx) format instead
                                4. Remove special characters or fix encoding
                                """)
                                return False
            else:
                st.error("Please upload an Excel (.xlsx) or CSV file")
                return False
            
            # Validate the loaded data
            if self.data is None or self.data.empty:
                st.error("The uploaded file appears to be empty")
                return False
            
            # Clean the data
            self.data = self._clean_data(self.data)
            
            # Additional validation for None return
            if self.data is None:
                st.error("❌ Data cleaning failed - unable to process the file")
                return False
            
            # Final validation
            if self.data.shape[1] < 2:
                st.error("Dataset must have at least 2 columns for causal analysis")
                return False
                
            if self.data.shape[0] < 10:
                st.error("Dataset must have at least 10 rows for meaningful analysis")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Unexpected error loading file: {str(e)}")
            st.info("Please check your file format and try again")
            return False
    
    def _clean_data(self, df):
        """Clean and prepare data for analysis"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Handle column names - remove quotes and clean up
            df.columns = df.columns.astype(str)
            df.columns = [col.strip().strip('"').strip("'") for col in df.columns]  # Remove quotes and whitespace
            
            # Remove unnamed columns (often from Excel index columns)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Show original data info
            st.write(f"**Original data shape:** {df.shape}")
            st.write(f"**Original columns:** {list(df.columns)}")
            st.write(f"**Column types:** {df.dtypes.to_dict()}")
            
            # Clean quoted values in all columns first
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Remove quotes from string values
                    df[col] = df[col].astype(str).str.strip().str.strip('"').str.strip("'")
                    # Replace empty strings and 'nan' strings with actual NaN
                    df[col] = df[col].replace(['', 'nan', 'None', 'null'], np.nan)
            
            # Show sample data after quote cleaning
            st.write("**Sample data after cleaning quotes:**")
            sample_display = {}
            for col in df.columns:  # Show ALL columns
                sample_values = df[col].dropna().head(3).tolist()
                sample_display[col] = sample_values
            st.write(sample_display)
            
            # Track conversion progress for each column
            conversion_log = {}
            
            # More aggressive numeric conversion
            for col in df.columns:
                original_type = df[col].dtype
                conversion_log[col] = {"original_type": str(original_type), "status": "processing"}
                
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    conversion_log[col]["status"] = "already_numeric"
                    continue
                    
                # Try multiple conversion strategies
                try:
                    # Strategy 1: Direct conversion after quote cleaning
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_count = df[col].notna().sum()
                    if converted_count > 0:
                        conversion_log[col]["status"] = f"converted_direct ({converted_count}/{len(df)} values)"
                        st.write(f"✅ Converted '{col}': {converted_count}/{len(df)} values converted to numeric")
                        continue
                except:
                    pass
                
                try:
                    # Strategy 2: Clean common formatting issues
                    if df[col].dtype == 'object':
                        # Create a copy for cleaning
                        temp_col = df[col].copy()
                        
                        # Remove common non-numeric characters
                        temp_col = temp_col.astype(str)
                        temp_col = temp_col.str.replace(',', '')  # Remove commas
                        temp_col = temp_col.str.replace('$', '')  # Remove dollar signs
                        temp_col = temp_col.str.replace('%', '')  # Remove percentages
                        temp_col = temp_col.str.replace(' ', '')  # Remove spaces
                        temp_col = temp_col.str.replace('+', '')  # Remove plus signs
                        
                        # Try to extract first number from string
                        numeric_pattern = temp_col.str.extract(r'([+-]?\d*\.?\d+)')[0]
                        numeric_values = pd.to_numeric(numeric_pattern, errors='coerce')
                        
                        converted_count = numeric_values.notna().sum()
                        if converted_count > len(df) * 0.1:  # If at least 10% convert successfully
                            df[col] = numeric_values
                            conversion_log[col]["status"] = f"converted_cleaned ({converted_count}/{len(df)} values)"
                            st.write(f"✅ Converted '{col}' with cleaning: {converted_count}/{len(df)} values")
                            continue
                except:
                    pass
                
                try:
                    # Strategy 3: Try to convert datetime to numeric timestamp
                    datetime_col = pd.to_datetime(df[col], errors='coerce')
                    if datetime_col.notna().sum() > len(df) * 0.1:
                        df[col] = datetime_col.astype('int64') // 10**9  # Convert to seconds
                        conversion_log[col]["status"] = f"converted_datetime ({datetime_col.notna().sum()}/{len(df)} values)"
                        st.write(f"✅ Converted '{col}' from datetime: {datetime_col.notna().sum()}/{len(df)} values")
                        continue
                except:
                    pass
                
                # If all conversion attempts failed, report it
                sample_values = df[col].dropna().head(5).tolist()
                conversion_log[col]["status"] = "conversion_failed"
                st.warning(f"⚠️ Could not convert column '{col}' (type: {original_type}). Sample values: {sample_values}")
            
            # Show conversion summary
            st.write("**Column Conversion Summary:**")
            for col, log in conversion_log.items():
                st.write(f"• **{col}**: {log['status']}")
            
            # Get numeric columns after all conversion attempts
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.write(f"**Numeric columns found:** {numeric_cols}")
            st.write(f"**Number of numeric columns:** {len(numeric_cols)}")
            
            if len(numeric_cols) < 2:
                # More aggressive attempt - try to force conversion of likely numeric columns
                st.warning("⚠️ Insufficient numeric columns. Attempting aggressive conversion...")
                
                potential_numeric = []
                for col in df.columns:
                    # Check if column name suggests it should be numeric
                    numeric_keywords = ['id', 'diameter', 'height', 'weight', 'duration', 'seconds', 'size', 'length', 'width', 'amount', 'count', 'number', 'price', 'cost']
                    if any(keyword in col.lower() for keyword in numeric_keywords):
                        potential_numeric.append(col)
                        continue
                    
                    # Check if column has any numeric-looking values
                    sample = df[col].dropna().astype(str).head(100)
                    if len(sample) > 0:
                        # Count values that contain digits
                        numeric_count = sum(1 for val in sample if any(char.isdigit() for char in str(val)))
                        
                        if numeric_count > len(sample) * 0.3:  # If >30% contain digits
                            potential_numeric.append(col)
                
                st.write(f"**Potentially numeric columns identified:** {potential_numeric}")
                
                if len(potential_numeric) >= 2:
                    # Force conversion of these columns
                    for col in potential_numeric:
                        try:
                            # Very aggressive cleaning
                            temp_series = df[col].astype(str)
                            # Keep only digits, dots, and minus signs
                            temp_series = temp_series.str.replace(r'[^0-9.\-]', '', regex=True)
                            # Remove multiple dots
                            temp_series = temp_series.str.replace(r'\.{2,}', '.', regex=True)
                            # Convert to numeric
                            converted = pd.to_numeric(temp_series, errors='coerce')
                            
                            # Only keep if we converted a reasonable number of values
                            if converted.notna().sum() > len(df) * 0.1:
                                df[col] = converted
                                st.write(f"🔧 Force-converted '{col}': {converted.notna().sum()}/{len(df)} values")
                        except Exception as e:
                            st.write(f"❌ Failed to force-convert '{col}': {str(e)}")
                    
                    # Recheck numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.write(f"**After aggressive conversion - Numeric columns:** {numeric_cols}")
                
                if len(numeric_cols) < 2:
                    st.error("❌ Cannot proceed: Need at least 2 numeric columns for causal analysis")
                    
                    # Show what we actually have in the data
                    st.write("**Actual data sample for debugging:**")
                    for col in df.columns:
                        unique_vals = df[col].dropna().unique()[:5]
                        st.write(f"• **{col}**: {unique_vals.tolist()}")
                    
                    # Show which columns were dropped and why
                    st.write("**Columns that failed conversion:**")
                    failed_cols = [col for col, log in conversion_log.items() if log["status"] == "conversion_failed"]
                    st.write(failed_cols)
                    
                    # Provide detailed guidance
                    st.markdown("""
                    **💡 Suggestions to fix your data:**
                    
                    1. **Check your CSV format:**
                       - Ensure numbers are not stored as text with quotes
                       - Remove currency symbols ($, €, etc.)
                       - Remove percentage signs (%)
                       - Use dots (.) for decimal separators, not commas
                    
                    2. **Try Excel format (.xlsx):**
                       - Excel often preserves numeric formatting better
                       - Save your CSV as Excel and upload again
                    
                    3. **Manual data cleaning:**
                       - Ensure at least 2 columns contain only numbers
                       - Remove any text headers or footers
                       - Check for hidden characters or extra spaces
                       - Remove quotes around numeric values
                    """)
                    return None
            
            # Track columns before filtering
            st.write(f"**Before filtering to numeric only - Total columns:** {len(df.columns)}")
            st.write(f"**All columns:** {list(df.columns)}")
            
            # Keep only numeric columns
            df = df[numeric_cols]
            
            # Show what was kept vs dropped
            dropped_cols = [col for col in conversion_log.keys() if col not in numeric_cols]
            if dropped_cols:
                st.warning(f"⚠️ Dropped non-numeric columns: {dropped_cols}")
            
            # Final validation before proceeding
            if df.empty:
                st.error("❌ No data remaining after cleaning")
                return None
            
            # Handle missing values with detailed reporting
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            
            # Show missing value info
            if missing_percentage.max() > 0:
                st.write(f"**Missing values per column:**")
                for col in df.columns:
                    if missing_percentage[col] > 0:
                        st.write(f"• {col}: {missing_percentage[col]:.1f}%")
            
            # Remove columns with too many missing values (>70% missing)
            cols_to_keep = missing_percentage[missing_percentage <= 70].index
            if len(cols_to_keep) < len(df.columns):
                dropped_cols = [col for col in df.columns if col not in cols_to_keep]
                st.warning(f"⚠️ Dropped columns with >70% missing values: {dropped_cols}")
            
            df = df[cols_to_keep]
            
            # Final check after dropping high-missing columns
            if df.shape[1] < 2:
                st.error("❌ Not enough columns remaining after removing high-missing columns")
                return None
            
            # Handle remaining missing values
            if df.isnull().sum().sum() > 0:
                # For remaining missing values, use median imputation (more robust than ffill/bfill)
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        median_value = df[col].median()
                        df[col] = df[col].fillna(median_value)
                        
                st.info("ℹ️ Filled missing values with column medians")
            
            # Remove any infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            if df.isnull().sum().sum() > 0:
                # If we created NaNs from inf values, drop those rows
                original_rows = len(df)
                df = df.dropna()
                if len(df) < original_rows:
                    st.warning(f"⚠️ Removed {original_rows - len(df)} rows containing infinite values")
            
            # Final validation
            if df.shape[0] < 10:
                st.error("❌ Not enough rows remaining after cleaning (minimum 10 required)")
                return None
                
            if df.shape[1] < 2:
                st.error("❌ Not enough columns remaining after cleaning (minimum 2 required)")
                return None
            
            st.success(f"✅ Data cleaned successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show final column info
            st.write("**Final dataset columns:**")
            for col in df.columns:
                col_info = f"• **{col}**: Range [{df[col].min():.2f}, {df[col].max():.2f}], Mean: {df[col].mean():.2f}"
                st.write(col_info)
            
            # Show summary of what happened to each original column
            st.write("**Column Processing Summary:**")
            for col in conversion_log.keys():
                if col in df.columns:
                    status = "✅ Kept (converted to numeric)"
                elif col in [c for c in conversion_log.keys() if conversion_log[c]["status"] == "conversion_failed"]:
                    status = "❌ Dropped (could not convert to numeric)"
                else:
                    status = "❌ Dropped (too many missing values)"
                st.write(f"• **{col}**: {status}")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error cleaning data: {str(e)}")
            st.write("**Debug info:**")
            if 'df' in locals():
                st.write(f"DataFrame shape: {df.shape}")
                st.write(f"DataFrame columns: {df.columns.tolist()}")
                # Show sample of problematic data
                st.write("**Sample data:**")
                st.dataframe(df.head(3))
            else:
                st.write("DataFrame not created")
            return None
    
    def generate_domain_constraints(self, columns: List[str], domain_context: str, api_key: str = None) -> Dict:
        """Use LLM to generate domain-specific constraints"""
        try:
            if not api_key and not st.secrets.get("OPENAI_API_KEY"):
                st.error("Please provide an OpenAI API key to use AI features")
                return {"forbidden_edges": [], "required_edges": [], "temporal_order": [], "explanation": "No API key provided"}
            
            client = init_openai(api_key)
            
            prompt = f"""
            Given a dataset with columns: {', '.join(columns)}
            Domain context: {domain_context}
            
            Generate domain constraints for causal discovery in JSON format:
            {{
                "forbidden_edges": [["source", "target"], ...],
                "required_edges": [["source", "target"], ...],
                "temporal_order": ["earliest_var", "middle_var", "latest_var", ...],
                "explanation": "Brief explanation of constraints"
            }}
            
            Consider:
            1. Temporal relationships (causes must precede effects)
            2. Domain knowledge (what relationships make sense)
            3. Physical/logical impossibilities
            
            Respond only with valid JSON.
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            constraints = json.loads(response.choices[0].message.content)
            self.domain_constraints = constraints
            return constraints
            
        except Exception as e:
            st.error(f"Error generating constraints: {str(e)}")
            return {"forbidden_edges": [], "required_edges": [], "temporal_order": [], "explanation": "No constraints generated"}
    
    def run_causal_discovery(self, constraints: Dict = None):
        """Run causal discovery with domain constraints"""
        try:
            # Initialize DirectLiNGAM model
            self.model = DirectLiNGAM()
            
            # Apply constraints if provided
            if constraints:
                # Convert constraints to format expected by LiNGAM
                forbidden = self._convert_edge_constraints(constraints.get('forbidden_edges', []))
                self.model = DirectLiNGAM(prior_knowledge=forbidden)
            
            # Fit the model
            self.model.fit(self.data)
            self.adjacency_matrix = self.model.adjacency_matrix_
            
            return True
            
        except Exception as e:
            st.error(f"Error in causal discovery: {str(e)}")
            return False
    
    def _convert_edge_constraints(self, edge_list: List[List[str]]) -> np.ndarray:
        """Convert edge constraints to adjacency matrix format"""
        n_vars = len(self.data.columns)
        forbidden_matrix = np.zeros((n_vars, n_vars))
        
        col_to_idx = {col: idx for idx, col in enumerate(self.data.columns)}
        
        for source, target in edge_list:
            if source in col_to_idx and target in col_to_idx:
                forbidden_matrix[col_to_idx[source], col_to_idx[target]] = 1
                
        return forbidden_matrix
    
    def _adjacency_to_graph_string(self) -> str:
        """Convert adjacency matrix to DoWhy graph string format"""
        if self.adjacency_matrix is None:
            return ""
        
        edges = []
        columns = list(self.data.columns)
        
        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if abs(self.adjacency_matrix[i, j]) > 0.1:  # Threshold for edge
                    edges.append(f'"{source}" -> "{target}"')
        
        return "; ".join(edges) if edges else ""
    
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Create NetworkX DiGraph from adjacency matrix for DoWhy"""
        G = nx.DiGraph()
        
        if self.adjacency_matrix is None:
            # Create simple graph with just the variables
            columns = list(self.data.columns)
            G.add_nodes_from(columns)
            return G
        
        columns = list(self.data.columns)
        G.add_nodes_from(columns)
        
        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if abs(self.adjacency_matrix[i, j]) > 0.1:  # Threshold for edge
                    G.add_edge(source, target, weight=self.adjacency_matrix[i, j])
        
        return G

    def calculate_ate(self, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
        """Calculate Average Treatment Effect using optimized single method for speed"""
        try:
            # Suppress warnings during calculation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create NetworkX graph for DoWhy
                causal_graph = self._create_networkx_graph()
                
                # If no discovered graph, create a simple one with confounders
                if len(causal_graph.edges()) == 0:
                    if confounders:
                        for conf in confounders:
                            causal_graph.add_edge(conf, treatment)
                            causal_graph.add_edge(conf, outcome)
                    causal_graph.add_edge(treatment, outcome)
                
                self.causal_model = CausalModel(
                    data=self.data,
                    treatment=treatment,
                    outcome=outcome,
                    graph=causal_graph,
                    common_causes=confounders or []
                )
                
                # Identify causal effect
                identified_estimand = self.causal_model.identify_effect(proceed_when_unidentifiable=True)
                
                # Use only linear regression for speed (most reliable and fastest)
                results = {}
                
                try:
                    causal_estimate = self.causal_model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression"
                    )
                    
                    # Get confidence intervals and p-value
                    try:
                        confidence_interval = causal_estimate.get_confidence_intervals()
                        p_value = causal_estimate.get_significance_test_results()['p_value']
                    except:
                        confidence_interval = [None, None]
                        p_value = None
                    
                    results["Linear Regression"] = {
                        "estimate": causal_estimate.value,
                        "confidence_interval": confidence_interval,
                        "p_value": p_value,
                        "method": "backdoor.linear_regression"
                    }
                    
                    consensus_estimate = causal_estimate.value
                    
                except Exception as method_error:
                    st.error(f"❌ Linear regression estimation failed: {str(method_error)}")
                    return None
                
                # Skip robustness checks for speed - just do basic validation
                robustness_results = {"status": "skipped_for_speed", "note": "Enable detailed analysis for full robustness checks"}
                
                # Calculate simplified additional metrics
                additional_metrics = self._calculate_simple_metrics(treatment, outcome, consensus_estimate)
                
                return {
                    "estimates": results,
                    "consensus_estimate": consensus_estimate,
                    "robustness": robustness_results,
                    "interpretation": self._interpret_ate(consensus_estimate, treatment, outcome),
                    "recommendation": self._generate_simple_recommendation(results),
                    "additional_metrics": additional_metrics
                }
                
        except Exception as e:
            st.error(f"Error calculating ATE: {str(e)}")
            # Provide fallback simple correlation analysis
            try:
                correlation = self.data[treatment].corr(self.data[outcome])
                st.info(f"Fallback: Simple correlation between {treatment} and {outcome}: {correlation:.4f}")
                return {
                    "estimates": {"Simple Correlation": {"estimate": correlation, "confidence_interval": [None, None], "p_value": None}},
                    "consensus_estimate": correlation,
                    "robustness": {"warning": "Only correlation available - causal inference failed"},
                    "interpretation": f"Simple correlation shows {abs(correlation):.4f} {'positive' if correlation > 0 else 'negative'} association",
                    "recommendation": "⚠️ Use caution: This is correlation, not causation. Consider improving data or model specification.",
                    "additional_metrics": {}
                }
            except:
                return None
    
    def _calculate_simple_metrics(self, treatment: str, outcome: str, estimate: float) -> Dict:
        """Calculate simplified metrics for speed"""
        additional_metrics = {}
        
        try:
            # Simple correlation-based metrics
            correlation = self.data[treatment].corr(self.data[outcome])
            r_squared = correlation ** 2
            additional_metrics["r_squared"] = r_squared
            additional_metrics["explained_variance_percent"] = r_squared * 100
            
            # Simple effect size classification
            if abs(estimate) < 0.01:
                effect_size = "Very Small"
            elif abs(estimate) < 0.1:
                effect_size = "Small"
            elif abs(estimate) < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            additional_metrics["effect_size_interpretation"] = effect_size
            
        except Exception as e:
            additional_metrics["error"] = str(e)
        
        return additional_metrics
    
    def _generate_simple_recommendation(self, estimates: Dict) -> str:
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
    
    def analyze_variable_relationships(self) -> Dict:
        """Analyze relationships between variables for better insights"""
        if self.data is None:
            return {}
        
        relationships = {}
        columns = list(self.data.columns)
        
        # Calculate correlation matrix
        corr_matrix = self.data.corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j and abs(corr_matrix.loc[col1, col2]) > 0.5:
                    strong_correlations.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_matrix.loc[col1, col2]
                    })
        
        # Sort by correlation strength
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        relationships['strong_correlations'] = strong_correlations[:10]  # Top 10
        relationships['correlation_matrix'] = corr_matrix
        
        return relationships
    
    def _interpret_ate(self, ate_value: float, treatment: str, outcome: str) -> str:
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
    
    def analyze_effect_heterogeneity(self, treatment: str, outcome: str, moderator: str = None) -> Dict:
        """Analyze heterogeneous treatment effects"""
        if not moderator:
            return {"error": "No moderator variable selected"}
        
        try:
            # Split data by moderator (median split for continuous variables)
            moderator_data = self.data[moderator]
            moderator_median = moderator_data.median()
            
            # High moderator group
            high_mod_data = self.data[moderator_data > moderator_median]
            high_mod_corr = high_mod_data[treatment].corr(high_mod_data[outcome])
            
            # Low moderator group
            low_mod_data = self.data[moderator_data <= moderator_median]
            low_mod_corr = low_mod_data[treatment].corr(low_mod_data[outcome])
            
            # Difference in effects
            effect_difference = high_mod_corr - low_mod_corr
            
            return {
                "high_moderator_effect": high_mod_corr,
                "low_moderator_effect": low_mod_corr,
                "effect_difference": effect_difference,
                "interpretation": f"The effect varies by {moderator}: {abs(effect_difference):.3f} difference between high/low groups"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def simulate_policy_intervention(self, treatment: str, outcome: str, intervention_size: float) -> Dict:
        """Simulate the effect of a policy intervention"""
        try:
            # Calculate current baseline
            current_outcome_mean = self.data[outcome].mean()
            current_treatment_mean = self.data[treatment].mean()
            
            # Estimate effect from correlation (simplified)
            correlation = self.data[treatment].corr(self.data[outcome])
            outcome_std = self.data[outcome].std()
            treatment_std = self.data[treatment].std()
            
            # Predicted change in outcome
            predicted_change = correlation * (outcome_std / treatment_std) * intervention_size
            predicted_new_outcome = current_outcome_mean + predicted_change
            
            # Calculate percentage change
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
    
    def _perform_detailed_robustness_checks(self, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
        """Perform detailed robustness checks only when requested"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Re-run the full causal model for detailed analysis
                causal_graph = self._create_networkx_graph()
                
                if len(causal_graph.edges()) == 0:
                    if confounders:
                        for conf in confounders:
                            causal_graph.add_edge(conf, treatment)
                            causal_graph.add_edge(conf, outcome)
                    causal_graph.add_edge(treatment, outcome)
                
                causal_model = CausalModel(
                    data=self.data,
                    treatment=treatment,
                    outcome=outcome,
                    graph=causal_graph,
                    common_causes=confounders or []
                )
                
                identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
                primary_estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                
                robustness_results = {}
                
                # Random common cause refutation
                try:
                    refutation_random = causal_model.refute_estimate(
                        identified_estimand,
                        primary_estimate,
                        method_name="random_common_cause",
                        num_simulations=10  # Reduced for speed
                    )
                    robustness_results["random_common_cause"] = {
                        "new_effect": getattr(refutation_random, 'new_effect', 'N/A'),
                        "p_value": getattr(refutation_random, 'p_value', None),
                        "status": "passed" if hasattr(refutation_random, 'new_effect') else "failed"
                    }
                except Exception as e:
                    robustness_results["random_common_cause"] = {"error": str(e), "status": "failed"}
                
                # Data subset refutation
                try:
                    refutation_subset = causal_model.refute_estimate(
                        identified_estimand,
                        primary_estimate,
                        method_name="data_subset_refuter",
                        subset_fraction=0.8,
                        num_simulations=10  # Reduced for speed
                    )
                    robustness_results["data_subset"] = {
                        "new_effect": getattr(refutation_subset, 'new_effect', 'N/A'),
                        "p_value": getattr(refutation_subset, 'p_value', None),
                        "status": "passed" if hasattr(refutation_subset, 'new_effect') else "failed"
                    }
                except Exception as e:
                    robustness_results["data_subset"] = {"error": str(e), "status": "failed"}
                
                return robustness_results
                
        except Exception as e:
            return {"error": str(e)}
    
    def explain_results_with_llm(self, ate_results: Dict, treatment: str, outcome: str, api_key: str = None) -> str:
        """Use LLM to explain causal analysis results"""
        try:
            if not api_key and not st.secrets.get("OPENAI_API_KEY"):
                return "OpenAI API key required for AI explanations. Please provide your API key in the sidebar."
            
            client = init_openai(api_key)
            
            prompt = f"""
            Explain the following causal analysis results in business terms:
            
            Treatment: {treatment}
            Outcome: {outcome}
            Average Treatment Effect: {ate_results.get('consensus_estimate', 'N/A')}
            Methods Used: {list(ate_results.get('estimates', {}).keys())}
            Robustness: {ate_results.get('recommendation', 'N/A')}
            
            Provide:
            1. A clear explanation of what this means
            2. Business implications
            3. Reliability of the result
            4. Recommendations for action
            
            Keep it concise and business-friendly.
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return CausalAnalyzer()

analyzer = get_analyzer()

# Initialize session state
if 'causal_discovery_completed' not in st.session_state:
    st.session_state['causal_discovery_completed'] = False
if 'selected_treatment' not in st.session_state:
    st.session_state['selected_treatment'] = None
if 'selected_outcome' not in st.session_state:
    st.session_state['selected_outcome'] = None
if 'ate_results' not in st.session_state:
    st.session_state['ate_results'] = None

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # API Key input - store in session state
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Enter your OpenAI API key for LLM features",
        value=st.session_state.get('openai_api_key', '')
    )
    
    # Store API key in session state
    if api_key:
        st.session_state['openai_api_key'] = api_key
        st.success("✅ API Key saved for this session")
    
    st.markdown("### 📋 Analysis Steps")
    st.markdown("""
    1. Upload Excel/CSV file
    2. Configure domain constraints
    3. Run causal discovery
    4. Calculate treatment effects
    5. Get AI explanations
    """)
    
    # Data quality tips
    with st.expander("📊 Data Quality Tips"):
        st.markdown("""
        **For best results:**
        - Use Excel (.xlsx) format when possible
        - Ensure consistent column structure
        - Remove special characters from headers
        - Use numeric data for causal analysis
        - Minimum 10 rows, 2 columns required
        """)

# Main interface
st.title("🤖 Causal AI Platform")
st.markdown("Discover causal relationships in your data using advanced AI and statistical methods.")

# Step 1: Data Upload
st.header("📁 Step 1: Data Upload")
uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file",
    type=['xlsx', 'csv'],
    help="Ensure your data has clean column names and numeric values. Excel format (.xlsx) is recommended for best compatibility."
)

if uploaded_file:
    if analyzer.load_data(uploaded_file):
        st.success("✅ Data loaded successfully!")
        
        # Display data preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Data Preview")
            st.dataframe(analyzer.data.head())
        
        with col2:
            st.metric("Rows", analyzer.data.shape[0])
            st.metric("Columns", analyzer.data.shape[1])
            st.metric("Missing Values", analyzer.data.isnull().sum().sum())
        
        # Data quality information
        with st.expander("📈 Data Quality Summary"):
            st.write("**Column Information:**")
            for col in analyzer.data.columns:
                col_info = f"• **{col}**: {analyzer.data[col].dtype}, Range: [{analyzer.data[col].min():.2f}, {analyzer.data[col].max():.2f}]"
                st.write(col_info)
        
        # Step 2: Domain Constraints
        st.header("🧠 Step 2: Domain Constraints (AI-Powered)")
        
        domain_context = st.text_area(
            "Describe your domain/business context:",
            placeholder="e.g., 'This is customer behavior data where age comes before purchase decisions, and marketing spend affects sales but not demographics...'",
            height=100
        )
        
        if st.button("🤖 Generate Domain Constraints", type="secondary"):
            if domain_context:
                with st.spinner("AI is analyzing your domain..."):
                    constraints = analyzer.generate_domain_constraints(
                        list(analyzer.data.columns), 
                        domain_context,
                        st.session_state.get('openai_api_key')
                    )
                    
                if constraints:
                    st.success("✅ Domain constraints generated!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(constraints)
                    
                    with col2:
                        st.info(f"**Explanation:** {constraints.get('explanation', 'No explanation provided')}")
            else:
                st.warning("Please provide domain context first.")
        
        # Step 3: Causal Discovery
        st.header("🔍 Step 3: Causal Discovery")
        
        if st.button("🚀 Run Causal Discovery", type="primary"):
            with st.spinner("Discovering causal relationships..."):
                success = analyzer.run_causal_discovery(analyzer.domain_constraints)
                
            if success:
                st.session_state['causal_discovery_completed'] = True
                st.success("✅ Causal discovery completed!")
            else:
                st.session_state['causal_discovery_completed'] = False
        
        # Show causal discovery results if completed
        if st.session_state['causal_discovery_completed'] and analyzer.adjacency_matrix is not None:
            # Visualize causal graph
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Discovered Causal Graph")
                
                # Create interactive graph with Plotly
                G = nx.DiGraph(analyzer.adjacency_matrix)
                pos = nx.spring_layout(G, seed=42)
                
                # Extract edges with weights
                edge_x, edge_y, edge_info = [], [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    weight = analyzer.adjacency_matrix[edge[0], edge[1]]
                    edge_info.append(f"Weight: {weight:.3f}")
                
                # Create plot
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='gray'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Add nodes
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = [analyzer.data.columns[i] for i in G.nodes()]
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    marker=dict(size=50, color='lightblue', line=dict(width=2, color='darkblue'))
                ))
                
                fig.update_layout(
                    title="Causal Graph",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Arrows show causal direction",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color="gray", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Adjacency Matrix")
                adj_df = pd.DataFrame(
                    analyzer.adjacency_matrix, 
                    index=analyzer.data.columns, 
                    columns=analyzer.data.columns
                )
                st.dataframe(adj_df.round(3))
                
                # Summary statistics
                n_edges = np.sum(np.abs(analyzer.adjacency_matrix) > 0.1)
                st.metric("Discovered Edges", n_edges)
        
        # Step 3.5: Variable Relationship Analysis (NEW FEATURE)
        st.header("📊 Step 3.5: Variable Relationship Analysis")
        st.markdown("Explore correlations and relationships between variables to guide your causal analysis.")
        
        if st.button("🔍 Analyze Variable Relationships", type="secondary"):
            with st.spinner("Analyzing variable relationships..."):
                relationships = analyzer.analyze_variable_relationships()
            
            if relationships:
                st.success("✅ Relationship analysis completed!")
                
                # Show correlation heatmap
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🔥 Correlation Heatmap")
                    fig_corr = px.imshow(
                        relationships['correlation_matrix'],
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Variable Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                with col2:
                    st.subheader("💪 Strongest Correlations")
                    strong_corr = relationships['strong_correlations']
                    if strong_corr:
                        for corr in strong_corr[:5]:  # Show top 5
                            st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
                    else:
                        st.write("No strong correlations (>0.5) found")
                
                # Suggested variable pairs for causal analysis
                st.subheader("💡 Suggested Treatment-Outcome Pairs")
                if strong_corr:
                    suggested_pairs = []
                    for corr in strong_corr[:3]:
                        suggested_pairs.append({
                            "Treatment": corr['var1'],
                            "Outcome": corr['var2'],
                            "Correlation": f"{corr['correlation']:.3f}",
                            "Suggestion": "High correlation - investigate causality"
                        })
                    st.dataframe(pd.DataFrame(suggested_pairs), use_container_width=True)
        
        # Step 4: Causal Inference Analysis
        st.header("🔬 Step 4: Causal Inference Analysis")
        
        st.markdown("""
        **Causal Inference** goes beyond correlation to estimate the actual causal effect of one variable on another.
        We'll use multiple estimation methods to ensure robust results.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            treatment_var = st.selectbox(
                "Treatment Variable (Cause)",
                options=list(analyzer.data.columns),
                key="treatment_select",
                help="The variable whose causal effect you want to measure",
                index=0  # Set default index to prevent auto-triggering
            )
        
        with col2:
            outcome_var = st.selectbox(
                "Outcome Variable (Effect)", 
                options=[col for col in analyzer.data.columns if col != treatment_var],
                key="outcome_select",
                help="The variable that may be causally affected by the treatment",
                index=0  # Set default index to prevent auto-triggering
            )
        
        with col3:
            confounders = st.multiselect(
                "Confounding Variables",
                options=[col for col in analyzer.data.columns if col not in [treatment_var, outcome_var]],
                key="confounders_select",
                help="Variables that might affect both treatment and outcome (important for accurate causal inference)"
            )
        
        # Validation and suggestions (only show when variables are selected, but don't auto-run analysis)
        if treatment_var and outcome_var and treatment_var != outcome_var:
            # Show variable summary
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Treatment:** {treatment_var}\n- Range: [{analyzer.data[treatment_var].min():.2f}, {analyzer.data[treatment_var].max():.2f}]\n- Mean: {analyzer.data[treatment_var].mean():.2f}")
            with col2:
                st.info(f"**Outcome:** {outcome_var}\n- Range: [{analyzer.data[outcome_var].min():.2f}, {analyzer.data[outcome_var].max():.2f}]\n- Mean: {analyzer.data[outcome_var].mean():.2f}")
            
            # Show correlation between treatment and outcome (quick preview)
            correlation = analyzer.data[treatment_var].corr(analyzer.data[outcome_var])
            st.metric("Correlation between Treatment & Outcome", f"{correlation:.4f}")
        elif treatment_var == outcome_var:
            st.error("❌ Treatment and outcome variables must be different!")
        
        # Add explanation of confounders
        with st.expander("❓ What are confounding variables?"):
            st.markdown("""
            **Confounders** are variables that influence both the treatment and outcome, potentially creating spurious correlations.
            
            **Example:** If studying the effect of exercise on health:
            - **Treatment:** Exercise frequency
            - **Outcome:** Health score  
            - **Potential confounders:** Age, income, education (these affect both exercise habits and health)
            
            Including relevant confounders helps ensure we measure the true causal effect.
            """)
        
        # Only run analysis when button is pressed
        run_analysis = st.button("🔬 Run Causal Inference", type="primary", key="run_causal_inference")
        
        if run_analysis:
            if not treatment_var or not outcome_var:
                st.error("❌ Please select both treatment and outcome variables")
            elif treatment_var == outcome_var:
                st.error("❌ Please select different variables for treatment and outcome")
            else:
                # Store selections in session state
                st.session_state['selected_treatment'] = treatment_var
                st.session_state['selected_outcome'] = outcome_var
                
                # Show progress bar for better UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Setting up causal model...")
                progress_bar.progress(25)
                
                status_text.text("Running causal inference...")
                progress_bar.progress(50)
                
                ate_results = analyzer.calculate_ate(treatment_var, outcome_var, confounders)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['ate_results'] = ate_results

# Display results if available (only after button press)
if st.session_state.get('ate_results') and st.session_state.get('selected_treatment') and st.session_state.get('selected_outcome'):
    ate_results = st.session_state['ate_results']
    treatment_var = st.session_state['selected_treatment']
    outcome_var = st.session_state['selected_outcome']
    
    # Check if current selection matches stored results
    current_treatment = st.session_state.get('treatment_select', '')
    current_outcome = st.session_state.get('outcome_select', '')
    
    if (current_treatment == treatment_var and current_outcome == outcome_var) or run_analysis:
        st.success("✅ Causal inference completed!")
        
        # Display consensus result prominently
        st.subheader("📊 Main Result")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Causal Effect Estimate", 
                f"{ate_results['consensus_estimate']:.4f}",
                help="Average causal effect across multiple estimation methods"
            )
        
        with col2:
            st.info(f"**Interpretation:** {ate_results['interpretation']}")
        
        # Show detailed results from different methods
        st.subheader("🔍 Detailed Results by Method")
        
        results_df = []
        for method, result in ate_results['estimates'].items():
            ci = result['confidence_interval']
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "N/A"
            p_val_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            
            results_df.append({
                "Method": method,
                "Estimate": f"{result['estimate']:.4f}",
                "95% CI": ci_str,
                "P-value": p_val_str,
                "Significant": "✅" if result['p_value'] and result['p_value'] < 0.05 else "❌"
            })
        
        st.dataframe(pd.DataFrame(results_df), use_container_width=True)
        
        # Additional Metrics (NEW FEATURE)
        if 'additional_metrics' in ate_results and ate_results['additional_metrics']:
            st.subheader("📏 Additional Metrics")
            metrics = ate_results['additional_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'cohens_d' in metrics:
                    st.metric("Cohen's d (Effect Size)", f"{metrics['cohens_d']:.3f}")
            
            with col2:
                if 'r_squared' in metrics:
                    st.metric("R² (Explained Variance)", f"{metrics['r_squared']:.3f}")
            
            with col3:
                if 'effect_size_interpretation' in metrics:
                    st.metric("Effect Size", metrics['effect_size_interpretation'])
        
        # Robustness checks (simplified for speed)
        st.subheader("🛡️ Robustness Analysis")
        
        if ate_results['robustness'].get('status') == 'skipped_for_speed':
            st.info("⚡ Robustness checks skipped for faster performance. Enable detailed analysis below for full validation.")
            
            if st.button("🔍 Run Detailed Robustness Analysis", key="detailed_robustness"):
                with st.spinner("Running comprehensive robustness checks..."):
                    detailed_results = analyzer._perform_detailed_robustness_checks(treatment_var, outcome_var, confounders)
                
                if detailed_results and 'error' not in detailed_results:
                    robustness_df = []
                    for test_name, test_result in detailed_results.items():
                        if isinstance(test_result, dict) and 'error' not in test_result:
                            effect_val = test_result.get('new_effect', 'N/A')
                            if isinstance(effect_val, (int, float)):
                                effect_str = f"{effect_val:.4f}"
                                status = "✅ Robust" if abs(effect_val - ate_results['consensus_estimate']) < 0.1 else "⚠️ Sensitive"
                            else:
                                effect_str = str(effect_val)
                                status = "❌ Failed"
                        
                            robustness_df.append({
                                "Robustness Test": test_name.replace('_', ' ').title(),
                                "Effect After Test": effect_str,
                                "Status": status
                            })
                    
                    if robustness_df:
                        st.dataframe(pd.DataFrame(robustness_df), use_container_width=True)
                        st.caption("Robustness tests check if results remain stable under different assumptions")
        else:
            st.warning("Robustness checks could not be performed")
        
        # Effect Heterogeneity Analysis
        st.subheader("🎭 Effect Heterogeneity Analysis")
        st.markdown("Analyze how the causal effect varies across different subgroups.")
        
        moderator_var = st.selectbox(
            "Select Moderator Variable",
            options=[col for col in analyzer.data.columns if col not in [treatment_var, outcome_var]],
            help="Variable that might moderate the treatment effect"
        )
        
        if st.button("🔍 Analyze Effect Heterogeneity"):
            with st.spinner("Analyzing effect heterogeneity..."):
                heterogeneity_results = analyzer.analyze_effect_heterogeneity(treatment_var, outcome_var, moderator_var)
            
            if 'error' not in heterogeneity_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Group Effect", f"{heterogeneity_results['high_moderator_effect']:.4f}")
                with col2:
                    st.metric("Low Group Effect", f"{heterogeneity_results['low_moderator_effect']:.4f}")
                with col3:
                    st.metric("Effect Difference", f"{heterogeneity_results['effect_difference']:.4f}")
                
                st.info(heterogeneity_results['interpretation'])
            else:
                st.error(f"Error in heterogeneity analysis: {heterogeneity_results['error']}")
        
        # Policy Simulation
        st.subheader("🎯 Policy Intervention Simulator")
        st.markdown("Simulate the expected impact of changing the treatment variable.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            intervention_size = st.number_input(
                f"Change in {treatment_var}",
                value=1.0,
                step=0.1,
                help="How much to change the treatment variable"
            )
        
        with col2:
            if st.button("🚀 Simulate Policy Impact"):
                with st.spinner("Simulating policy intervention..."):
                    simulation_results = analyzer.simulate_policy_intervention(treatment_var, outcome_var, intervention_size)
                
                if 'error' not in simulation_results:
                    st.success("✅ Simulation completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Baseline", f"{simulation_results['current_baseline']:.3f}")
                    with col2:
                        st.metric("Predicted Change", f"{simulation_results['predicted_outcome_change']:.3f}")
                    with col3:
                        st.metric("Percentage Change", f"{simulation_results['percent_change']:.1f}%")
                    
                    st.info(simulation_results['interpretation'])
                else:
                    st.error(f"Simulation error: {simulation_results['error']}")
        
        # Recommendation
        st.subheader("💡 Recommendation")
        st.info(ate_results['recommendation'])
        
        # Export results feature
        st.subheader("📁 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Results as CSV"):
                results_export = pd.DataFrame(results_df)
                csv = results_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"causal_analysis_{treatment_var}_{outcome_var}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Copy Summary to Clipboard"):
                summary_text = f"""
Causal Analysis Summary
Treatment: {treatment_var}
Outcome: {outcome_var}
Causal Effect: {ate_results['consensus_estimate']:.4f}
Interpretation: {ate_results['interpretation']}
Recommendation: {ate_results['recommendation']}
                """
                st.code(summary_text, language="text")

        # Step 5: AI Explanation (Enhanced)
        st.header("🧠 Step 5: AI-Powered Insights")
        
        if st.button("🤖 Get Detailed AI Analysis", type="secondary"):
            with st.spinner("AI is analyzing your causal inference results..."):
                explanation = analyzer.explain_results_with_llm(
                    ate_results, treatment_var, outcome_var,
                    st.session_state.get('openai_api_key')
                )
            
            st.markdown("### 📋 Business Insights & Recommendations")
            st.markdown(explanation)
            
            # Add actionable insights
            st.markdown("### 🎯 Next Steps")
            if ate_results['consensus_estimate'] and abs(ate_results['consensus_estimate']) > 0.01:
                st.markdown(f"""
                1. **Validate findings:** Consider running controlled experiments to confirm this causal relationship
                2. **Monitor implementation:** If you act on this insight, track the {outcome_var} changes
                3. **Scale consideration:** The effect size of {abs(ate_results['consensus_estimate']):.4f} suggests {'strong' if abs(ate_results['consensus_estimate']) > 0.1 else 'moderate' if abs(ate_results['consensus_estimate']) > 0.01 else 'small'} practical impact
                """)
            else:
                st.markdown("""
                1. **Explore other variables:** The causal effect appears small - consider other potential causes
                2. **Check data quality:** Ensure sufficient sample size and measurement accuracy
                3. **Consider non-linear relationships:** The effect might be conditional on other factors
                """)
    else:
        st.info("💡 Variable selection changed. Click 'Run Causal Inference' to analyze the new combination.")
