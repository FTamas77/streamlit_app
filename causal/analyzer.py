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
            # Get the original numeric columns
            available_cols = list(self.discovery.numeric_columns)
            
            # Add encoded categorical columns if they exist
            if hasattr(self.discovery, 'encoded_data') and self.discovery.encoded_data is not None:
                encoded_cols = list(self.discovery.encoded_data.columns)
                for col in encoded_cols:
                    if col.endswith('_Code') and col not in available_cols:
                        available_cols.append(col)            
            return available_cols
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
        """Load data from uploaded Excel file with robust error handling and validation"""
        if uploaded_file is None:
            st.error("No file provided")
            return False
            
        # File size validation (limit to 50MB)
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:
            st.error("File is too large (>50MB). Please use a smaller file.")
            return False
            
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
            
            # Enhanced data validation
            if self.data is None:
                st.error("Failed to read the file - it may be corrupted or in an unsupported format")
                return False
                
            if self.data.empty:
                st.error("The uploaded file is empty")
                return False
            
            # Check for minimum data quality requirements
            if self.data.shape[1] < 2:
                st.error("Dataset must have at least 2 columns for causal analysis")
                return False
                
            if self.data.shape[0] < 10:
                st.error(f"Dataset has only {self.data.shape[0]} rows. At least 10 rows required for meaningful analysis")
                return False
            
            # Check for reasonable data size limits
            if self.data.shape[0] > 100000:
                st.warning(f"Large dataset detected ({self.data.shape[0]} rows). Processing may be slow.")
            
            if self.data.shape[1] > 500:
                st.warning(f"Many columns detected ({self.data.shape[1]}). Consider reducing dimensionality for better performance.")
            
            # Check data quality issues
            total_cells = self.data.shape[0] * self.data.shape[1]
            null_cells = self.data.isnull().sum().sum()
            null_percentage = (null_cells / total_cells) * 100
            
            if null_percentage > 50:
                st.error(f"Data has {null_percentage:.1f}% missing values, which is too high for reliable analysis")
                return False
            elif null_percentage > 20:
                st.warning(f"Data has {null_percentage:.1f}% missing values. Consider data cleaning.")
            
            # Check for duplicate columns
            duplicate_cols = self.data.columns[self.data.columns.duplicated()].tolist()
            if duplicate_cols:
                st.warning(f"Duplicate column names found: {duplicate_cols}. These will be renamed automatically.")
                self.data.columns = pd.io.common.dedup_names(self.data.columns, is_potential_multiindex=False)
            
            # Check for empty columns
            empty_cols = [col for col in self.data.columns if self.data[col].isnull().all()]
            if empty_cols:
                st.warning(f"Empty columns found and will be removed: {empty_cols}")
                self.data = self.data.drop(columns=empty_cols)
            
            # Check for columns with only one unique value (excluding nulls)
            constant_cols = []
            for col in self.data.columns:
                unique_vals = self.data[col].dropna().nunique()
                if unique_vals <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                st.warning(f"Constant columns found (will be problematic for causal analysis): {constant_cols}")
            
            # Import data cleaning function
            try:
                from utils.data_cleaning import clean_data
                original_shape = self.data.shape
                self.data = clean_data(self.data)
                
                if self.data is None:
                    st.error("❌ Data cleaning failed - unable to process the file")
                    return False
                
                if self.data.shape != original_shape:
                    st.info(f"Data cleaned: {original_shape} → {self.data.shape}")
                    
            except ImportError:
                st.warning("Data cleaning module not available - using raw data")
            except Exception as e:
                st.warning(f"Data cleaning failed: {str(e)} - using raw data")
            
            # Final validation after cleaning
            if self.data.shape[1] < 2:
                st.error("After cleaning, dataset has fewer than 2 columns")
                return False
                
            if self.data.shape[0] < 10:
                st.error(f"After cleaning, dataset has only {self.data.shape[0]} rows")
                return False
            
            # Check for numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning(f"Only {len(numeric_cols)} numeric columns found. Causal analysis requires at least 2 numeric variables.")
            
            st.success(f"✅ Data loaded successfully: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            
            return True            
        except Exception as e:
            st.error(f"Unexpected error loading file: {str(e)}")
            st.info("Please check your file format and try again")
            return False    
    def run_causal_discovery(self, constraints: Dict = None):
        """Run causal discovery with domain constraints and validation"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data before running causal discovery.")
        
        if self.data.empty:
            raise ValueError("Data is empty. Cannot perform causal discovery.")
        
        try:
            success = self.discovery.run_discovery(self.data, constraints)
            if success:
                self.adjacency_matrix = self.discovery.get_adjacency_matrix()
                if self.adjacency_matrix is None:
                    st.warning("Causal discovery completed but no adjacency matrix was generated")
            return success
        except Exception as e:
            st.error(f"Causal discovery failed: {str(e)}")
            return False

    def calculate_ate(self, treatment: str, outcome: str) -> Dict:
        """
        Calculate Average Treatment Effect using DoWhy with comprehensive validation.
        
        Note: This method requires causal discovery to have been run first.
        Confounders are automatically identified from the causal graph structure.
        
        Args:
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable
        
        Returns:
            Dict containing causal effect estimates and metadata
            
        Raises:
            ValueError: If inputs are invalid, data is insufficient, or causal discovery not run
            ImportError: If DoWhy is not available
            RuntimeError: If causal inference fails
        """        # Pre-validation checks
        if not DOWHY_AVAILABLE:
            raise ImportError("DoWhy is required for ATE calculation but not available. Please install DoWhy.")
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data before calculating ATE.")
        
        if self.data.empty:
            raise ValueError("Data is empty. Cannot calculate ATE.")        # Check if causal discovery has been run
        if self.adjacency_matrix is None:
            raise ValueError("Causal discovery must be run before causal inference. Please run causal discovery first.")
        
        # Input validation
        if not treatment or not isinstance(treatment, str):
            raise ValueError(f"Treatment must be a non-empty string, got: {treatment}")
        
        if not outcome or not isinstance(outcome, str):
            raise ValueError(f"Outcome must be a non-empty string, got: {outcome}")
        
        if treatment == outcome:
            raise ValueError("Treatment and outcome variables cannot be the same")        
        # Get available columns for validation
        available_columns = set(self.data.columns)
        if hasattr(self.discovery, 'encoded_data') and self.discovery.encoded_data is not None:
            available_columns.update(self.discovery.encoded_data.columns)
        
        # Check if variables exist in available data
        missing_vars = []
        if treatment not in available_columns:
            missing_vars.append(f"treatment '{treatment}'")
        if outcome not in available_columns:
            missing_vars.append(f"outcome '{outcome}'")
        
        if missing_vars:
            available_list = sorted(list(available_columns))
            raise ValueError(f"Variables not found: {', '.join(missing_vars)}. Available columns: {available_list}")
        
        # Data quality checks
        try:
            # Check if we have sufficient data for the analysis
            analysis_cols = [treatment, outcome]
            
            # Determine which dataset to check
            data_to_check = self.data
            if (hasattr(self.discovery, 'encoded_data') and self.discovery.encoded_data is not None and
                any(col.endswith('_Code') for col in analysis_cols)):
                data_to_check = self.discovery.encoded_data
            
            # Check if all required columns exist in the selected dataset
            missing_in_selected = [col for col in analysis_cols if col not in data_to_check.columns]
            if missing_in_selected:
                # Try the other dataset
                if data_to_check is self.data and hasattr(self.discovery, 'encoded_data'):
                    data_to_check = self.discovery.encoded_data
                elif data_to_check is not self.data:
                    data_to_check = self.data
                
                missing_in_selected = [col for col in analysis_cols if col not in data_to_check.columns]
                if missing_in_selected:
                    raise ValueError(f"Variables not found in any dataset: {missing_in_selected}")
            
            # Check data completeness
            analysis_data = data_to_check[analysis_cols]
            complete_rows = analysis_data.dropna()
            
            if len(complete_rows) < 10:
                raise ValueError(f"Insufficient data: only {len(complete_rows)} complete rows available (minimum 10 required)")
            
            if len(complete_rows) < len(analysis_data) * 0.5:
                st.warning(f"High missing data: {len(analysis_data) - len(complete_rows)} rows have missing values")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            else:
                raise RuntimeError(f"Data validation failed: {str(e)}") from e
          # All validation passed - proceed with ATE calculation
        try:
            from causal.inference import calculate_ate_dowhy
            result = calculate_ate_dowhy(self, treatment, outcome)
            
            # Additional result validation
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid result type returned: {type(result)}")
            
            if 'consensus_estimate' not in result:
                st.warning("Result missing consensus estimate - this may indicate calculation issues")
            
            return result
            
        except Exception as e:
            # Re-raise known exceptions
            if isinstance(e, (ValueError, ImportError)):
                raise
            else:
                # Wrap unexpected exceptions
                raise RuntimeError(f"ATE calculation failed: {str(e)}") from e
    
    def analyze_variable_relationships(self) -> Dict:
        """Analyze relationships between variables for better insights"""
        if self.data is None:
            return {}
        
        from analytics.relationship_analysis import analyze_relationships
        return analyze_relationships(self.data)
