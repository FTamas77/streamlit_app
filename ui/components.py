import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from llm.llm import explain_results_with_llm

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
    """Display interactive causal graph with directional arrows
    
    Parameters:
    - adjacency_matrix (np.ndarray): MUTABLE - 2D array representing causal relationships
    - columns (List[str]): MUTABLE - List of column names for graph labels
    """
    import numpy as np
    
    # Create NetworkX directed graph from adjacency matrix
    G = nx.DiGraph()
    
    # Add nodes with column names
    for i, col in enumerate(columns):
        G.add_node(i, label=col)
    
    # Add edges based on adjacency matrix (only if weight > threshold)
    threshold = 0.01
    for i in range(len(columns)):
        for j in range(len(columns)):
            if abs(adjacency_matrix[i, j]) > threshold:
                # DirectLiNGAM: adjacency_matrix[i,j] represents effect from j to i
                G.add_edge(j, i, weight=adjacency_matrix[i, j])
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
    
    # Create plot
    fig = go.Figure()
    
    # Add edges with arrows
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Calculate arrow position (90% along the edge)
        arrow_x = x0 + 0.9 * (x1 - x0)
        arrow_y = y0 + 0.9 * (y1 - y0)
        
        # Calculate arrow direction
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
          # Edge line (consistent width for all edges)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=2, color='gray'),
            hoverinfo='text',
            hovertext=f"{columns[edge[0]]} → {columns[edge[1]]}<br>Weight: {weight:.3f}",
            mode='lines',
            showlegend=False
        ))
        
        # Arrow head
        arrow_size = 0.03
        arrow_x1 = arrow_x - arrow_size * (dx + dy)
        arrow_y1 = arrow_y - arrow_size * (dy - dx)
        arrow_x2 = arrow_x - arrow_size * (dx - dy)
        arrow_y2 = arrow_y - arrow_size * (dy + dx)
        
        fig.add_trace(go.Scatter(
            x=[arrow_x1, arrow_x, arrow_x2],
            y=[arrow_y1, arrow_y, arrow_y2],
            line=dict(width=2, color='darkgray'),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [columns[i] for i in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=[f"Variable: {text}" for text in node_text],
        text=node_text,
        textposition="middle center",
        textfont=dict(size=12, color='darkblue'),
        marker=dict(
            size=60, 
            color='lightblue', 
            line=dict(width=2, color='darkblue')
        ),
        showlegend=False
    ))
    
    # Add title with edge count
    edge_count = len(G.edges())
    title_text = f"Causal Graph ({edge_count} causal relationships)"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=500
    )
    
    if edge_count == 0:
        st.info("ℹ️ No causal relationships detected (all weights below threshold)")
    else:
        st.info(f"📊 Showing {edge_count} causal relationships. Hover over edges to see weights.")
    
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
