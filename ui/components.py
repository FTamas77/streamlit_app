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
    """Display highly interactive causal graph with modern web features
    
    Parameters:
    - adjacency_matrix (np.ndarray): MUTABLE - 2D array representing causal relationships
      DirectLiNGAM format: adjacency_matrix[i,j] = effect strength from variable j to variable i
    - columns (List[str]): MUTABLE - List of column names for graph labels
    - column_mapping (Dict): Optional mapping from encoded names to original info for display
    
    Features:
    - Interactive node selection and highlighting
    - Dynamic edge filtering by strength
    - Zoom and pan capabilities
    - Hover tooltips with detailed information
    - Node clustering and layout options
    - Export capabilities
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import networkx as nx
    import colorsys
    
    # Create user-friendly display names
    display_names = create_display_names(columns, column_mapping)
    
    # Calculate edge statistics for better visualization
    edge_weights = []
    valid_edges = []
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            if abs(adjacency_matrix[i, j]) > 0.01:
                edge_weights.append(abs(adjacency_matrix[i, j]))
                valid_edges.append((j, i, adjacency_matrix[i, j]))
    
    if not valid_edges:
        st.info("ℹ️ No causal relationships detected (all weights below threshold)")
        return
    
    # Create NetworkX graph with enhanced attributes
    G = nx.DiGraph()
    
    # Add nodes with enhanced attributes
    for i, display_name in enumerate(display_names):
        # Calculate node importance (sum of incoming and outgoing edges)
        importance = sum(abs(adjacency_matrix[i, :]) + abs(adjacency_matrix[:, i]))
        
        G.add_node(i, 
                  label=display_name,
                  original_name=columns[i],
                  importance=importance,
                  in_degree=0,
                  out_degree=0)
    
    # Add edges with detailed attributes
    edge_strengths = [abs(weight) for _, _, weight in valid_edges]
    min_strength, max_strength = min(edge_strengths), max(edge_strengths)
    
    for source, target, weight in valid_edges:
        normalized_strength = (abs(weight) - min_strength) / (max_strength - min_strength) if max_strength > min_strength else 0.5
        
        G.add_edge(source, target, 
                  weight=weight,
                  abs_weight=abs(weight),
                  normalized_strength=normalized_strength,
                  edge_type='strong' if abs(weight) > 0.5 else 'medium' if abs(weight) > 0.2 else 'weak')
        
        # Update node degrees
        G.nodes[source]['out_degree'] += 1
        G.nodes[target]['in_degree'] += 1
    
    # Enhanced layout options
    layout_options = {
        'spring': nx.spring_layout(G, seed=42, k=3, iterations=100),
        'circular': nx.circular_layout(G),
        'kamada_kawai': nx.kamada_kawai_layout(G) if len(G.nodes) > 2 else nx.spring_layout(G, seed=42),
        'shell': nx.shell_layout(G) if len(G.nodes) > 3 else nx.spring_layout(G, seed=42)
    }
    
    # UI Controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        layout_choice = st.selectbox(
            "📐 Layout Style", 
            options=['spring', 'circular', 'kamada_kawai', 'shell'],
            index=0,
            help="Choose how nodes are arranged"
        )
    
    with col2:
        min_edge_strength = st.slider(
            "🔍 Min Edge Strength", 
            min_value=0.0, 
            max_value=max(edge_strengths) if edge_strengths else 1.0,
            value=0.01,
            step=0.01,
            help="Filter edges by minimum strength"
        )
    
    with col3:
        show_edge_labels = st.checkbox("🏷️ Show Edge Labels", value=False, help="Display edge weights on graph")
    
    with col4:
        node_size_factor = st.slider("📏 Node Size", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    # Filter edges based on strength
    filtered_edges = [(s, t, w) for s, t, w in valid_edges if abs(w) >= min_edge_strength]
    
    if not filtered_edges:
        st.warning(f"⚠️ No edges meet the minimum strength threshold of {min_edge_strength:.3f}")
        return
    
    # Get selected layout
    pos = layout_options[layout_choice]
    
    # Create the interactive plot
    fig = go.Figure()
    
    # Add edges with enhanced styling
    edge_traces = []
    edge_label_traces = []
    
    for source, target, weight in filtered_edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Calculate edge properties
        abs_weight = abs(weight)
        normalized_strength = (abs_weight - min_strength) / (max_strength - min_strength) if max_strength > min_strength else 0.5
        
        # Dynamic edge styling
        if abs_weight > 0.5:
            edge_color = f'rgba(214, 39, 40, {0.6 + 0.4 * normalized_strength})'  # Red with alpha
            edge_width = 4 + 2 * normalized_strength
        elif abs_weight > 0.2:
            edge_color = f'rgba(255, 127, 14, {0.5 + 0.3 * normalized_strength})'  # Orange with alpha
            edge_width = 3 + 1 * normalized_strength
        else:
            edge_color = f'rgba(31, 119, 180, {0.4 + 0.2 * normalized_strength})'  # Blue with alpha
            edge_width = 2 + 0.5 * normalized_strength
        
        # Calculate arrow positioning
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Adjust for node size
            node_radius = 0.08 * node_size_factor
            start_x = x0 + node_radius * dx_norm
            start_y = y0 + node_radius * dy_norm
            end_x = x1 - node_radius * dx_norm
            end_y = y1 - node_radius * dy_norm
            
            # Edge line
            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode='lines',
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='text',
                hovertext=f"""
                <b>{display_names[source]} → {display_names[target]}</b><br>
                Weight: {weight:.4f}<br>
                Strength: {abs_weight:.4f}<br>
                Type: {'Strong' if abs_weight > 0.5 else 'Medium' if abs_weight > 0.2 else 'Weak'}<br>
                Direction: {'Positive' if weight > 0 else 'Negative'} influence
                """,
                showlegend=False,
                name=f"Edge_{source}_{target}"
            ))
            
            # Enhanced arrow head
            arrow_pos = 0.85
            arrow_x = start_x + arrow_pos * (end_x - start_x)
            arrow_y = start_y + arrow_pos * (end_y - start_y)
            
            arrow_size = 0.04 * (1 + normalized_strength) * node_size_factor
            arrow_x1 = arrow_x - arrow_size * (dx_norm + dy_norm)
            arrow_y1 = arrow_y - arrow_size * (dy_norm - dx_norm)
            arrow_x2 = arrow_x - arrow_size * (dx_norm - dy_norm)
            arrow_y2 = arrow_y - arrow_size * (dy_norm + dx_norm)
            
            fig.add_trace(go.Scatter(
                x=[arrow_x1, arrow_x, arrow_x2, arrow_x1],
                y=[arrow_y1, arrow_y, arrow_y2, arrow_y1],
                fill='toself',
                fillcolor=edge_color.replace('rgba', 'rgb').replace(', 0.', ', 1.').replace(', 1.', ', 1.0'),
                line=dict(width=1, color=edge_color),
                mode='lines',
                showlegend=False,
                hoverinfo='skip',
                name=f"Arrow_{source}_{target}"
            ))
            
            # Edge labels (optional)
            if show_edge_labels:
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                fig.add_trace(go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode='text',
                    text=[f'{weight:.2f}'],
                    textfont=dict(size=10, color='black'),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f"Label_{source}_{target}"
                ))
    
    # Add nodes with enhanced interactivity
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Dynamic node sizing based on importance
    node_sizes = []
    node_colors = []
    hover_texts = []
    
    for node in G.nodes():
        importance = G.nodes[node]['importance']
        in_degree = G.nodes[node]['in_degree']
        out_degree = G.nodes[node]['out_degree']
        
        # Size based on importance
        base_size = 60 * node_size_factor
        size_multiplier = 1 + (importance / max(1, max([G.nodes[n]['importance'] for n in G.nodes()])))
        node_size = base_size * size_multiplier
        node_sizes.append(node_size)
        
        # Color based on node role
        if in_degree > out_degree:
            # More incoming edges - likely an outcome
            node_color = '#FF6B6B'  # Coral red
        elif out_degree > in_degree:
            # More outgoing edges - likely a cause
            node_color = '#4ECDC4'  # Teal
        else:
            # Balanced - mediator
            node_color = '#45B7D1'  # Sky blue
        
        node_colors.append(node_color)
        
        # Enhanced hover text
        original_name = G.nodes[node]['original_name']
        display_name = G.nodes[node]['label']
        
        hover_text = f"""
        <b>{display_name}</b><br>
        {'Original: ' + original_name + '<br>' if original_name != display_name else ''}
        Role: {'Outcome' if in_degree > out_degree else 'Cause' if out_degree > in_degree else 'Mediator'}<br>
        Incoming edges: {in_degree}<br>
        Outgoing edges: {out_degree}<br>
        Importance score: {importance:.3f}
        """
        
        # Add encoding info if available
        if column_mapping and original_name in [col for col in columns if col.endswith('_Code')]:
            for encoded_col, mapping_info in column_mapping.items():
                if encoded_col == original_name:
                    hover_text += f"<br><br><b>Encoding:</b><br>"
                    for val, code in list(mapping_info['encoding'].items())[:3]:
                        hover_text += f"{val} → {code}<br>"
                    if len(mapping_info['encoding']) > 3:
                        hover_text += "..."
                    break
        
        hover_texts.append(hover_text)
    
    # Add interactive nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=3, color='white'),
            opacity=0.9
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=12, color='white', family='Arial Black'),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False,
        name="Nodes"
    ))
    
    # Enhanced layout with better controls
    fig.update_layout(
        title=dict(
            text=f"🔗 Interactive Causal Graph ({len(filtered_edges)} relationships)",
            x=0.5,
            font=dict(size=20, color='#2E86AB')
        ),
        showlegend=True,
        legend=dict(
            title="<b>Edge Strength</b>",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=20, r=120, t=80),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        height=700,
        # Enable zoom and pan
        dragmode='pan'
    )
    
    # Add legend traces for edge types
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='lines', 
                  line=dict(width=6, color='rgba(214, 39, 40, 0.8)'), 
                  name='Strong (>0.5)', showlegend=True),
        go.Scatter(x=[None], y=[None], mode='lines', 
                  line=dict(width=4, color='rgba(255, 127, 14, 0.7)'), 
                  name='Medium (0.2-0.5)', showlegend=True),
        go.Scatter(x=[None], y=[None], mode='lines', 
                  line=dict(width=2, color='rgba(31, 119, 180, 0.6)'), 
                  name='Weak (<0.2)', showlegend=True)
    ]
    
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Display the interactive graph
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
        'modeBarButtonsToRemove': ['autoScale2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'causal_graph',
            'height': 700,
            'width': 1200,
            'scale': 2
        }
    })
    
    # Graph statistics and insights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔗 Total Edges", len(filtered_edges))
    
    with col2:
        avg_strength = np.mean([abs(w) for _, _, w in filtered_edges])
        st.metric("💪 Avg Strength", f"{avg_strength:.3f}")
    
    with col3:
        strong_edges = sum(1 for _, _, w in filtered_edges if abs(w) > 0.5)
        st.metric("🔴 Strong Edges", strong_edges)
    
    with col4:
        max_importance = max([G.nodes[n]['importance'] for n in G.nodes()])
        most_important = [display_names[n] for n in G.nodes() if G.nodes[n]['importance'] == max_importance][0]
        st.metric("⭐ Key Variable", most_important)
    
    # Interactive node analysis
    with st.expander("🔍 Detailed Node Analysis", expanded=False):
        selected_node = st.selectbox(
            "Select a variable to analyze:",
            options=display_names,
            help="Choose a variable to see its detailed causal relationships"
        )
        
        if selected_node:
            node_idx = display_names.index(selected_node)
            
            # Incoming edges (causes)
            incoming = [(display_names[j], adjacency_matrix[node_idx, j]) for j in range(len(columns)) 
                       if abs(adjacency_matrix[node_idx, j]) > 0.01]
            
            # Outgoing edges (effects)
            outgoing = [(display_names[i], adjacency_matrix[i, node_idx]) for i in range(len(columns))                       if abs(adjacency_matrix[i, node_idx]) > 0.01]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**📥 What influences {selected_node}:**")
                if incoming:
                    for cause, weight in sorted(incoming, key=lambda x: abs(x[1]), reverse=True):
                        direction = "🔴" if weight < 0 else "🟢"
                        st.write(f"{direction} **{cause}**: {weight:.4f}")
                else:
                    st.write("No direct causes detected")
            
            with col2:
                st.write(f"**📤 What {selected_node} influences:**")
                if outgoing:
                    for effect, weight in sorted(outgoing, key=lambda x: abs(x[1]), reverse=True):
                        direction = "🔴" if weight < 0 else "🟢"
                        st.write(f"{direction} **{effect}**: {weight:.4f}")
                else:
                    st.write("No direct effects detected")

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
