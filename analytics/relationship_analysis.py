from typing import Dict

def analyze_relationships(data) -> Dict:
    """Analyze relationships between variables for better insights"""
    relationships = {}
    columns = list(data.columns)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
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
