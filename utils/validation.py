"""
Input validation utilities for Causal AI Platform
Centralized validation logic for data integrity
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List
from config.constants import ALLOWED_FILE_TYPES, MAX_FILE_SIZE_MB

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_uploaded_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file for size, format, and basic structure
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ALLOWED_FILE_TYPES:
        return False, f"Unsupported file type. Please upload: {', '.join(ALLOWED_FILE_TYPES)}"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size: {MAX_FILE_SIZE_MB}MB"
    
    return True, None

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate DataFrame for causal analysis requirements
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "Dataset is empty"
    
    # Minimum rows check
    if len(df) < 10:
        return False, f"Dataset too small ({len(df)} rows). Minimum 10 rows required for reliable analysis"
    
    # Minimum columns check
    if len(df.columns) < 2:
        return False, f"Need at least 2 variables for causal analysis. Found {len(df.columns)} columns"
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        return False, f"Columns with all missing values: {', '.join(null_columns)}"
    
    # Check data variance (all constant values)
    constant_columns = []
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    
    if len(constant_columns) == len(df.select_dtypes(include=['number']).columns):
        return False, "All numeric columns have constant values. Cannot perform analysis"
    
    return True, None

def validate_variable_selection(treatment: str, outcome: str, available_vars: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate treatment and outcome variable selection
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not treatment or not outcome:
        return False, "Please select both treatment and outcome variables"
    
    if treatment == outcome:
        return False, "Treatment and outcome variables must be different"
    
    if treatment not in available_vars:
        return False, f"Treatment variable '{treatment}' not found in dataset"
    
    if outcome not in available_vars:
        return False, f"Outcome variable '{outcome}' not found in dataset"
    
    return True, None

def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate OpenAI API key format
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    if not api_key.startswith('sk-'):
        return False, "Invalid API key format. Should start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "API key appears too short"
    
    return True, None

def safe_execute(func, error_message: str = "An error occurred"):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        error_message: Custom error message to display
    
    Returns:
        Result of function or None if error occurred
    """
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return None
