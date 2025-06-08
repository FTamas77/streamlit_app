import pandas as pd
import numpy as np
import warnings
from typing import Dict, List
import streamlit as st

# Try to import optional dependencies
try:
    from lingam import DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

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
                self.data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                # Try multiple CSV reading strategies
                try:
                    self.data = pd.read_csv(uploaded_file, encoding='utf-8')
                except (pd.errors.ParserError, UnicodeDecodeError) as e:
                    st.warning(f"Standard CSV parsing failed: {str(e)}. Trying alternative methods...")
                    uploaded_file.seek(0)
                    
                    try:
                        self.data = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.info("✅ Successfully loaded with latin-1 encoding")
                    except Exception:
                        uploaded_file.seek(0)
                        try:
                            self.data = pd.read_csv(
                                uploaded_file, 
                                encoding='utf-8',
                                error_bad_lines=False,
                                warn_bad_lines=True,
                                on_bad_lines='skip'
                            )
                            st.warning("⚠️ Some rows were skipped due to formatting issues")
                        except Exception:
                            uploaded_file.seek(0)
                            try:
                                self.data = pd.read_csv(
                                    uploaded_file,
                                    engine='python',
                                    encoding='utf-8',
                                    sep=None,
                                    skipinitialspace=True,
                                    quoting=3
                                )
                                st.info("✅ Successfully loaded with flexible parsing")
                            except Exception as final_error:
                                st.error(f"Unable to parse CSV file after multiple attempts: {str(final_error)}")
                                return False
            else:
                st.error("Please upload an Excel (.xlsx) or CSV file")
                return False
            
            if self.data is None or self.data.empty:
                st.error("The uploaded file appears to be empty")
                return False
            
            # Import data cleaning function
            from utils.data_cleaning import clean_data
            self.data = clean_data(self.data)
            
            if self.data is None:
                st.error("❌ Data cleaning failed - unable to process the file")
                return False
            
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
    
    def run_causal_discovery(self, constraints: Dict = None):
        """Run causal discovery with domain constraints"""
        if not LINGAM_AVAILABLE:
            raise ImportError("LiNGAM is required for causal discovery but not available")
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print(f"DEBUG: Running DirectLiNGAM on data shape: {self.data.shape}")
                print(f"DEBUG: Data columns: {list(self.data.columns)}")
                
                if constraints:
                    from utils.constraint_utils import convert_edge_constraints
                    
                    # DirectLiNGAM only supports forbidden edges via prior_knowledge
                    forbidden = convert_edge_constraints(constraints.get('forbidden_edges', []), self.data.columns)
                    print(f"DEBUG: Using constraints - forbidden matrix shape: {forbidden.shape}")
                    
                    self.model = DirectLiNGAM(prior_knowledge=forbidden)
                else:
                    print("DEBUG: Running DirectLiNGAM without constraints")
                    self.model = DirectLiNGAM()
                
                self.model.fit(self.data)
                self.adjacency_matrix = self.model.adjacency_matrix_
                
                print(f"DEBUG: DirectLiNGAM produced adjacency matrix shape: {self.adjacency_matrix.shape}")
                print(f"DEBUG: DirectLiNGAM adjacency matrix:\n{self.adjacency_matrix}")
                
                # Check if matrix is lower triangular (proper causal order)
                is_lower_triangular = np.allclose(self.adjacency_matrix, np.tril(self.adjacency_matrix))
                print(f"DEBUG: Matrix is lower triangular (proper causal order): {is_lower_triangular}")
                
                # Check causal order
                if hasattr(self.model, 'causal_order_'):
                    print(f"DEBUG: Causal order: {self.model.causal_order_}")
                    causal_order_vars = [self.data.columns[i] for i in self.model.causal_order_]
                    print(f"DEBUG: Causal order variables: {causal_order_vars}")
                
                # Count non-zero edges
                non_zero_edges = np.count_nonzero(np.abs(self.adjacency_matrix) > 0.01)
                print(f"DEBUG: Number of edges with |weight| > 0.01: {non_zero_edges}")
                
                # Enforce required edges if specified
                if constraints and 'required_edges' in constraints:
                    self._enforce_required_edges(constraints['required_edges'])
            
            return True
            
        except Exception as e:
            print(f"ERROR in causal discovery: {str(e)}")
            return False

    def calculate_ate(self, treatment: str, outcome: str, confounders: List[str] = None) -> Dict:
        """
        Calculate Average Treatment Effect using DoWhy
        
        Args:
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable  
            confounders: List of confounder variable names (optional)
                        If provided, these will be used as adjustment variables.
                        If None, confounders will be identified from the graph structure.
        """
        if not DOWHY_AVAILABLE:
            raise ImportError("DoWhy is required for ATE calculation but not available")
            
        from calculations.causal_inference import calculate_ate_dowhy
        return calculate_ate_dowhy(self, treatment, outcome, confounders)
    
    def analyze_variable_relationships(self) -> Dict:
        """Analyze relationships between variables for better insights"""
        if self.data is None:
            return {}
        
        from calculations.relationship_analysis import analyze_relationships
        return analyze_relationships(self.data)
    
    def analyze_effect_heterogeneity(self, treatment: str, outcome: str, moderator: str = None) -> Dict:
        """Analyze heterogeneous treatment effects"""
        from calculations.advanced_analysis import analyze_heterogeneity
        return analyze_heterogeneity(self.data, treatment, outcome, moderator)
    
    def simulate_policy_intervention(self, treatment: str, outcome: str, intervention_size: float) -> Dict:
        """Simulate the effect of a policy intervention"""
        from calculations.advanced_analysis import simulate_intervention
        return simulate_intervention(self.data, treatment, outcome, intervention_size)
    
    def _enforce_required_edges(self, required_edges: List[List[str]]):
        """Post-process adjacency matrix to ensure required edges exist"""
        if self.adjacency_matrix is None:
            return
            
        columns = list(self.data.columns)
        for edge in required_edges:
            if len(edge) == 2:
                from_var, to_var = edge
                if from_var in columns and to_var in columns:
                    from_idx = columns.index(from_var)
                    to_idx = columns.index(to_var)
                    # Force this edge to exist with minimum threshold
                    if abs(self.adjacency_matrix[to_idx, from_idx]) < 0.1:
                        self.adjacency_matrix[to_idx, from_idx] = 0.1
