import numpy as np

def correlation_based_discovery(data) -> np.ndarray:
    """Fallback causal discovery using correlation analysis"""
    corr_matrix = data.corr().abs()
    n_vars = len(data.columns)
    
    # Create adjacency matrix based on strong correlations
    adjacency = np.zeros((n_vars, n_vars))
    threshold = 0.3  # Correlation threshold for edges
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and corr_matrix.iloc[i, j] > threshold:
                # Simple heuristic: higher variance variable influences lower variance
                var_i = data.iloc[:, i].var()
                var_j = data.iloc[:, j].var()
                if var_i > var_j:
                    adjacency[i, j] = corr_matrix.iloc[i, j]
                else:
                    adjacency[j, i] = corr_matrix.iloc[i, j]
    
    return adjacency
