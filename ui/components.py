import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from ai.llm_integration import explain_results_with_llm

def show_data_preview(data):
    """Display data preview with metrics
    
    Parameters:
    - data (pd.DataFrame): MUTABLE - Pandas DataFrame, passed by reference
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Data Preview")
        st.dataframe(data.head())
    
    with col2:
        st.metric("Rows", data.shape[0])
        st.metric("Columns", data.shape[1])
        st.metric("Missing Values", data.isnull().sum().sum())

def show_data_quality_summary(data):
    """Show data quality information
    
    Parameters:
    - data (pd.DataFrame): MUTABLE - Pandas DataFrame, passed by reference
    """
    with st.expander("📈 Data Quality Summary"):
        st.write("**Column Information:**")
        for col in data.columns:
            col_info = f"• **{col}**: {data[col].dtype}, Range: [{data[col].min():.2f}, {data[col].max():.2f}]"
            st.write(col_info)

def show_correlation_heatmap(correlation_matrix):
    """Display correlation heatmap
    
    Parameters:
    - correlation_matrix (pd.DataFrame or np.ndarray): MUTABLE - Correlation matrix data
    """
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Variable Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

def show_causal_graph(adjacency_matrix, columns):
    """Display interactive causal graph
    
    Parameters:
    - adjacency_matrix (np.ndarray): MUTABLE - 2D array representing causal relationships
    - columns (List[str]): MUTABLE - List of column names for graph labels
    """
    G = nx.DiGraph(adjacency_matrix)
    pos = nx.spring_layout(G, seed=42)
    
    # Extract edges with weights
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
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
    node_text = [columns[i] for i in G.nodes()]
    
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
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_results_table(ate_results):
    """Display results in a formatted table
    
    Parameters:
    - ate_results (Dict): MUTABLE - Dictionary containing analysis results, passed by reference
    """
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

def show_ai_enhanced_results(ate_results, treatment, outcome, api_key=None):
    """New function using AI integration from other file
    
    Parameters:
    - ate_results (Dict): MUTABLE - Analysis results dictionary
    - treatment (str): IMMUTABLE - Treatment variable name
    - outcome (str): IMMUTABLE - Outcome variable name  
    - api_key (str, optional): IMMUTABLE - API key for AI features
    """
    st.subheader("🤖 AI-Enhanced Analysis")
    
    # Use function from ai.llm_integration module
    ai_explanation = explain_results_with_llm(ate_results, treatment, outcome, api_key)
    st.write(ai_explanation)
