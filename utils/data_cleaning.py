import pandas as pd
import numpy as np
import streamlit as st

def clean_data(df):
    """Clean and preprocess the data with proper pandas handling"""
    if df is None or df.empty:
        return None
    
    try:
        # Make a proper copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle column names - remove quotes and clean up
        df.columns = df.columns.astype(str)
        df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Show original data info
        st.write(f"**Original data shape:** {df.shape}")
        st.write(f"**Original columns:** {list(df.columns)}")
        
        # Clean quoted values in all columns first
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.strip('"').str.strip("'")
                df[col] = df[col].replace(['', 'nan', 'None', 'null'], np.nan)
        
        # Convert to numeric
        df = _convert_to_numeric(df)
        
        if df is None:
            return None
        
        # Handle missing values
        df = _handle_missing_values(df)
        
        # Final validation
        if df.shape[0] < 10 or df.shape[1] < 2:
            st.error("❌ Not enough data remaining after cleaning")
            return None
        
        st.success(f"✅ Data cleaned successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        st.error(f"❌ Error cleaning data: {str(e)}")
        return None

def _convert_to_numeric(df):
    """Convert columns to numeric types"""
    conversion_log = {}
    
    for col in df.columns:
        original_type = df[col].dtype
        conversion_log[col] = {"original_type": str(original_type), "status": "processing"}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            conversion_log[col]["status"] = "already_numeric"
            continue
            
        # Try direct conversion
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted_count = df[col].notna().sum()
            if converted_count > 0:
                conversion_log[col]["status"] = f"converted_direct ({converted_count}/{len(df)} values)"
                continue
        except:
            pass
        
        # Try cleaning common formatting issues
        try:
            if df[col].dtype == 'object':
                temp_col = df[col].copy().astype(str)
                temp_col = temp_col.str.replace(',', '').str.replace('$', '').str.replace('%', '').str.replace(' ', '').str.replace('+', '')
                numeric_pattern = temp_col.str.extract(r'([+-]?\d*\.?\d+)')[0]
                numeric_values = pd.to_numeric(numeric_pattern, errors='coerce')
                
                converted_count = numeric_values.notna().sum()
                if converted_count > len(df) * 0.1:
                    df[col] = numeric_values
                    conversion_log[col]["status"] = f"converted_cleaned ({converted_count}/{len(df)} values)"
                    continue
        except:
            pass
        
        # Mark as failed
        conversion_log[col]["status"] = "conversion_failed"
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Cannot proceed: Need at least 2 numeric columns for causal analysis")
        return None
    
    # Keep only numeric columns
    df = df[numeric_cols]
    return df

def _handle_missing_values(df):
    """Handle missing values in the dataset"""
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Remove columns with too many missing values (>70% missing)
    cols_to_keep = missing_percentage[missing_percentage <= 70].index
    if len(cols_to_keep) < len(df.columns):
        dropped_cols = [col for col in df.columns if col not in cols_to_keep]
        st.warning(f"⚠️ Dropped columns with >70% missing values: {dropped_cols}")
    
    # Create a proper copy to avoid SettingWithCopyWarning
    df = df[cols_to_keep].copy()
    
    if df.shape[1] < 2:
        return None
    
    # Fill remaining missing values with median
    if df.isnull().sum().sum() > 0:
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        st.info("ℹ️ Filled missing values with column medians")
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isnull().sum().sum() > 0:
        original_rows = len(df)
        df = df.dropna()
        if len(df) < original_rows:
            st.warning(f"⚠️ Removed {original_rows - len(df)} rows containing infinite values")
    
    return df
