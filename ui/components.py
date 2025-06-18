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
            # Check if column is numeric
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # For numeric columns, show range with proper formatting
                try:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    col_info = f"• **{col}**: {data[col].dtype}, Range: [{min_val:.2f}, {max_val:.2f}]"
                except (ValueError, TypeError):
                    # Fallback if numeric operations fail
                    col_info = f"• **{col}**: {data[col].dtype}, Unique values: {data[col].nunique()}"
            else:
                # For categorical/text columns, show unique count and sample values
                unique_count = data[col].nunique()
                if unique_count <= 10:
                    # Show all unique values if there are few
                    unique_vals = list(data[col].unique())
                    col_info = f"• **{col}**: {data[col].dtype}, Values: {unique_vals}"
                else:
                    # Show count and first few values if there are many
                    sample_vals = list(data[col].unique()[:5])
                    col_info = f"• **{col}**: {data[col].dtype}, {unique_count} unique values (e.g., {sample_vals}...)"
            
            st.write(col_info)

def create_display_names(columns, column_mapping=None):
    """Create user-friendly display names for encoded variables
    
    Parameters:
    - columns (List[str]): List of column names (may include encoded names like 'Fuel_Type_Code')
    - column_mapping (Dict): Optional mapping from encoded names to original info
    
    Returns:
    - List[str]: User-friendly display names
    """
    if not column_mapping:
        return columns
    
    display_names = []
    for col in columns:
        if col in column_mapping:
            # Use original column name for display
            original_name = column_mapping[col]['original_column']
            display_names.append(original_name)
        else:
            # Keep original name
            display_names.append(col)
    
    return display_names

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

def show_causal_graph(adjacency_matrix, columns, column_mapping=None):
    """Display interactive causal graph with directional arrows
    
    Parameters:
    - adjacency_matrix (np.ndarray): MUTABLE - 2D array representing causal relationships
      DirectLiNGAM format: adjacency_matrix[i,j] = effect strength from variable j to variable i
    - columns (List[str]): MUTABLE - List of column names for graph labels
    - column_mapping (Dict): Optional mapping from encoded names to original info for display
    
    Note: Arrows point from cause to effect (j → i when adjacency_matrix[i,j] != 0)
    """
    import numpy as np
    
    # Create user-friendly display names
    display_names = create_display_names(columns, column_mapping)
    
    # Debug information for edge directions
    print(f"DEBUG: Creating causal graph with {len(columns)} nodes")
    edge_count = 0
    for i in range(len(columns)):
        for j in range(len(columns)):
            if abs(adjacency_matrix[i, j]) > 0.01:
                print(f"DEBUG: Edge {display_names[j]} → {display_names[i]} (weight: {adjacency_matrix[i, j]:.3f})")
                edge_count += 1
    print(f"DEBUG: Total edges to display: {edge_count}")
    
    # Create NetworkX directed graph from adjacency matrix
    G = nx.DiGraph()
    
    # Add nodes with display names
    for i, display_name in enumerate(display_names):
        G.add_node(i, label=display_name)
    
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
        
        # Calculate direction vector and normalize
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        # Calculate node radius (to avoid arrows going under nodes)
        node_radius = 0.08  # Approximate radius based on node size
        
        # Adjust start and end points to avoid overlap with nodes
        start_x = x0 + node_radius * dx
        start_y = y0 + node_radius * dy
        end_x = x1 - node_radius * dx
        end_y = y1 - node_radius * dy
          # Calculate arrow position (85% along the adjusted edge to avoid node overlap)
        arrow_x = start_x + 0.85 * (end_x - start_x)
        arrow_y = start_y + 0.85 * (end_y - start_y)
        
        # Edge line with better visibility
        # Use different colors for different edge strengths
        edge_strength = abs(weight)
        if edge_strength > 0.5:
            edge_color = '#d62728'  # Strong red for strong relationships
            edge_width = 4
        elif edge_strength > 0.2:
            edge_color = '#ff7f0e'  # Orange for medium relationships
            edge_width = 3
        else:
            edge_color = '#1f77b4'  # Blue for weak relationships
            edge_width = 2
        
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[start_y, end_y],  # Use adjusted coordinates
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='text',
            hovertext=f"{display_names[edge[0]]} → {display_names[edge[1]]}<br>Weight: {weight:.3f}<br>Strength: {'Strong' if edge_strength > 0.5 else 'Medium' if edge_strength > 0.2 else 'Weak'}",
            mode='lines',
            showlegend=False
        ))
        
        # Improved arrow head - larger and more visible
        arrow_size = 0.05  # Increased from 0.03
        arrow_x1 = arrow_x - arrow_size * (dx + dy)
        arrow_y1 = arrow_y - arrow_size * (dy - dx)
        arrow_x2 = arrow_x - arrow_size * (dx - dy)
        arrow_y2 = arrow_y - arrow_size * (dy + dx)
        
        # Arrow head with fill for better visibility
        fig.add_trace(go.Scatter(
            x=[arrow_x1, arrow_x, arrow_x2, arrow_x1],  # Close the triangle
            y=[arrow_y1, arrow_y, arrow_y2, arrow_y1],
            fill='toself',
            fillcolor=edge_color,
            line=dict(width=2, color=edge_color),
            mode='lines',
            showlegend=False,
            hoverinfo='skip'        ))
      # Add nodes with improved styling
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [display_names[i] for i in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=[f"Variable: {text}" for text in node_text],
        text=node_text,
        textposition="middle center",
        textfont=dict(size=12, color='darkblue', family='Arial Black'),  # Changed to dark blue for visibility
        marker=dict(
            size=80,  # Increased from 60 
            color='lightcyan',  # Light background for better text contrast
            line=dict(width=3, color='#2E86AB'),
            opacity=0.9
        ),
        showlegend=False
    ))
    
    # Add legend for edge colors
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers', 
                  marker=dict(size=10, color='#d62728'), 
                  name='Strong (>0.5)', showlegend=True),
        go.Scatter(x=[None], y=[None], mode='markers', 
                  marker=dict(size=10, color='#ff7f0e'), 
                  name='Medium (0.2-0.5)', showlegend=True),
        go.Scatter(x=[None], y=[None], mode='markers', 
                  marker=dict(size=10, color='#1f77b4'), 
                  name='Weak (<0.2)', showlegend=True)
    ]
    
    for trace in legend_traces:
        fig.add_trace(trace)    # Add title with edge count and better styling
    edge_count = len(G.edges())
    title_text = f"🔗 Causal Relationships Graph ({edge_count} connections)"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18, color='#2E86AB')),
        showlegend=True,
        legend=dict(
            title="Edge Strength:",
            orientation="v",  # Changed to vertical
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,  # Position to the right of the graph
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=80, t=60),  # Increased right margin for legend
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        height=600
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

        
def show_interactive_scenario_explorer(ate_results, treatment_var, outcome_var, analyzer):
    """Interactive scenario explorer with real-time visual feedback
    
    Parameters:
    - ate_results (Dict): ATE calculation results
    - treatment_var (str): Name of treatment variable  
    - outcome_var (str): Name of outcome variable
    - analyzer: CausalAnalyzer instance for recalculating scenarios
    """
    st.markdown("### 🎮 Interactive Scenario Explorer")
    st.write("**Adjust the sliders below to explore different policy scenarios and see the impact in real-time:**")
    
    # Safety check: ensure variables are numeric
    if not pd.api.types.is_numeric_dtype(analyzer.data[treatment_var]):
        st.error(f"❌ Treatment variable '{treatment_var}' is not numeric. Cannot run scenario explorer.")
        return
    
    if not pd.api.types.is_numeric_dtype(analyzer.data[outcome_var]):
        st.error(f"❌ Outcome variable '{outcome_var}' is not numeric. Cannot run scenario explorer.")
        return
    
    # Get current state
    current_treatment_mean = analyzer.data[treatment_var].mean()
    current_outcome_mean = analyzer.data[outcome_var].mean()
    ate_estimate = ate_results['consensus_estimate']
    
    # Create two columns for scenario controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Scenario 1: Increase Treatment")
        if analyzer.data[treatment_var].dtype in ['int64', 'bool']:
            # For binary treatment, show percentage of population treated
            increase_pct = st.slider(
                "Percentage of population to receive treatment",
                min_value=0.0, max_value=100.0, 
                value=100.0, step=5.0,
                key="increase_slider"
            ) / 100.0
            scenario1_treatment = increase_pct
            scenario1_label = f"{increase_pct*100:.0f}% treated"
        else:
            # For continuous treatment, show percentage increase
            increase_pct = st.slider(
                "Increase treatment by (%)",
                min_value=0, max_value=200, 
                value=50, step=10,
                key="increase_slider"
            )
            scenario1_treatment = current_treatment_mean * (1 + increase_pct/100)
            scenario1_label = f"+{increase_pct}% increase"
        
        scenario1_outcome = current_outcome_mean + ate_estimate * (scenario1_treatment - current_treatment_mean)
        scenario1_change = scenario1_outcome - current_outcome_mean
        
        # Display scenario 1 results
        st.metric(
            label=f"{treatment_var} (Scenario 1)",
            value=f"{scenario1_treatment:.3f}",
            delta=f"{scenario1_treatment - current_treatment_mean:+.3f}"
        )
        st.metric(
            label=f"{outcome_var} (Scenario 1)", 
            value=f"{scenario1_outcome:.3f}",
            delta=f"{scenario1_change:+.3f}"
        )
    
    with col2:
        st.markdown("#### 📉 Scenario 2: Decrease Treatment")
        if analyzer.data[treatment_var].dtype in ['int64', 'bool']:
            # For binary treatment, show percentage of population treated
            decrease_pct = st.slider(
                "Percentage of population to receive treatment ",
                min_value=0.0, max_value=100.0, 
                value=0.0, step=5.0,
                key="decrease_slider"
            ) / 100.0
            scenario2_treatment = decrease_pct
            scenario2_label = f"{decrease_pct*100:.0f}% treated"
        else:
            # For continuous treatment, show percentage decrease
            decrease_pct = st.slider(
                "Decrease treatment by (%)",
                min_value=0, max_value=100, 
                value=25, step=5,
                key="decrease_slider"
            )
            scenario2_treatment = current_treatment_mean * (1 - decrease_pct/100)
            scenario2_label = f"-{decrease_pct}% decrease"
        
        scenario2_outcome = current_outcome_mean + ate_estimate * (scenario2_treatment - current_treatment_mean)
        scenario2_change = scenario2_outcome - current_outcome_mean
        
        # Display scenario 2 results
        st.metric(
            label=f"{treatment_var} (Scenario 2)",
            value=f"{scenario2_treatment:.3f}",
            delta=f"{scenario2_treatment - current_treatment_mean:+.3f}"
        )
        st.metric(
            label=f"{outcome_var} (Scenario 2)",
            value=f"{scenario2_outcome:.3f}",
            delta=f"{scenario2_change:+.3f}"
        )
    
    # Visualization
    st.markdown("#### 📊 Visual Comparison")
    
    # Create comparison chart
    comparison_data = pd.DataFrame({
        'Scenario': ['Current State', scenario1_label, scenario2_label],
        treatment_var: [current_treatment_mean, scenario1_treatment, scenario2_treatment],
        outcome_var: [current_outcome_mean, scenario1_outcome, scenario2_outcome],
        'Change in Outcome': [0, scenario1_change, scenario2_change]
    })
    
    # Create bar chart showing outcome changes
    fig = px.bar(
        comparison_data, 
        x='Scenario', 
        y='Change in Outcome',
        color='Change in Outcome',
        color_continuous_scale='RdYlBu_r',
        title=f"Impact on {outcome_var} by Scenario",
        labels={'Change in Outcome': f'Change in {outcome_var}'}
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("#### 📋 Scenario Summary")
    summary_df = comparison_data.copy()
    for col in [treatment_var, outcome_var, 'Change in Outcome']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)
    st.dataframe(summary_df, use_container_width=True)
    
    # Business insights
    st.markdown("#### 💡 Key Insights")
    if abs(scenario1_change) > abs(scenario2_change):
        better_scenario = "Scenario 1" if scenario1_change > scenario2_change else "Scenario 2" 
        worse_scenario = "Scenario 2" if scenario1_change > scenario2_change else "Scenario 1"
        st.success(f"**{better_scenario}** shows the largest positive impact on {outcome_var}")
    
    if ate_estimate > 0:
        st.info(f"💡 **Interpretation:** Since ATE = {ate_estimate:.3f} > 0, increasing {treatment_var} generally improves {outcome_var}")
    elif ate_estimate < 0:
        st.info(f"💡 **Interpretation:** Since ATE = {ate_estimate:.3f} < 0, decreasing {treatment_var} generally improves {outcome_var}")
    else:
        st.warning(f"⚠️ **Interpretation:** ATE ≈ 0 suggests {treatment_var} has minimal impact on {outcome_var}")
