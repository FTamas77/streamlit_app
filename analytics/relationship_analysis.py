import pandas as pd
import numpy as np
from typing import Dict

def analyze_relationships(data) -> Dict:
    """Analyze relationships between variables for better insights"""
    relationships = {}
    
    # Filter to numeric columns only - correlation can't handle categorical data
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_columns = list(numeric_data.columns)
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"DEBUG: Analyzing relationships for {len(numeric_columns)} numeric columns")
    print(f"DEBUG: Excluded {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    if len(numeric_columns) < 2:
        print("WARNING: Need at least 2 numeric columns for correlation analysis")
        return {
            'strong_correlations': [],
            'correlation_matrix': pd.DataFrame(),
            'message': 'Insufficient numeric data for correlation analysis'
        }
    
    # Calculate correlation matrix on numeric data only
    corr_matrix = numeric_data.corr()
    
    # Find strongest correlations
    strong_correlations = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
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
    relationships['numeric_columns'] = numeric_columns
    relationships['categorical_columns'] = categorical_columns
    relationships['total_correlations_found'] = len(strong_correlations)
    
    print(f"DEBUG: Found {len(strong_correlations)} strong correlations (|r| > 0.5)")
    
    return relationships
