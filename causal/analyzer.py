import pandas as pd
import numpy as np
import warnings
from typing import Dict, List
import streamlit as st

# Import our new discovery module
from causal.discovery import CausalDiscovery

# Try to import optional dependencies
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

class CausalAnalyzer:
    """Main class for causal analysis pipeline - orchestrates discovery and inference"""
    
    def __init__(self):
        self.data = None
        self.discovery = CausalDiscovery()
        self.adjacency_matrix = None
        self.causal_model = None
        self.domain_constraints = None
    
    def get_numeric_columns(self):
        """Get list of numeric columns available for causal analysis"""
        if hasattr(self.discovery, 'numeric_columns') and self.discovery.numeric_columns:
            return self.discovery.numeric_columns
        elif self.data is not None:
            # Fallback: determine numeric columns from data
            return list(self.data.select_dtypes(include=[np.number]).columns)
        else:
            return []
    
    def get_categorical_columns(self):
        """Get list of categorical columns that were excluded from causal analysis"""
        if hasattr(self.discovery, 'categorical_columns') and self.discovery.categorical_columns:
            return self.discovery.categorical_columns
        elif self.data is not None:
            # Fallback: determine categorical columns from data
            return list(self.data.select_dtypes(exclude=[np.number]).columns)
        else:
            return []
        
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
        success = self.discovery.run_discovery(self.data, constraints)
        if success:
            self.adjacency_matrix = self.discovery.get_adjacency_matrix()
        return success

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
            
        from causal.inference import calculate_ate_dowhy
        return calculate_ate_dowhy(self, treatment, outcome, confounders)
    
    def analyze_variable_relationships(self) -> Dict:
        """Analyze relationships between variables for better insights"""
        if self.data is None:
            return {}
        
        from analytics.relationship_analysis import analyze_relationships
        return analyze_relationships(self.data)
