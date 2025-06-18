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
      DirectLiNGAM format: adjacency_matrix[i,j] = causal coefficient from variable j to variable i
    - columns (List[str]): MUTABLE - List of column names for graph labels
    - column_mapping (Dict): Optional mapping from encoded names to original info for display
    
    Features:
    - Interactive node selection and highlighting
    - Dynamic edge filtering by coefficient magnitude
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
    edge_magnitudes = []
    valid_edges = []
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            if abs(adjacency_matrix[i, j]) > 0.01:
                edge_magnitudes.append(abs(adjacency_matrix[i, j]))
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
                  out_degree=0)    # Add edges
    for source, target, weight in valid_edges:
        G.add_edge(source, target, weight=weight)
        
        # Update node degrees
        G.nodes[source]['out_degree'] += 1
        G.nodes[target]['in_degree'] += 1
      # Enhanced layout options with fallback handling
    layout_functions = {
        'spring': lambda G: nx.spring_layout(G, seed=42, k=2, iterations=50),
        'circular': lambda G: nx.circular_layout(G),
        'kamada_kawai': lambda G: nx.kamada_kawai_layout(G) if len(G.nodes) > 2 else nx.spring_layout(G, seed=42),
        'shell': lambda G: nx.shell_layout(G) if len(G.nodes) > 3 else nx.circular_layout(G),
        'random': lambda G: nx.random_layout(G, seed=42)
    }
    
    # Try to compute layouts with error handling
    layout_options = {}
    for name, func in layout_functions.items():
        try:
            layout_options[name] = func(G)
        except Exception as e:
            # Fallback to spring layout if any layout fails
            try:
                layout_options[name] = nx.spring_layout(G, seed=42)
            except:                # Ultimate fallback - manual positioning
                layout_options[name] = {i: (i * 0.2, 0) for i in range(len(G.nodes))}
      # UI Controls
    col1, col2 = st.columns([3, 2])
    
    with col1:
        layout_choice = st.selectbox(
            "📐 Graph Layout", 
            options=['spring', 'circular', 'kamada_kawai', 'shell', 'random'],
            index=0,
            help="Choose how nodes are arranged in the graph. Spring layout is usually best for small graphs."
        )
    
    with col2:
        node_size_factor = st.slider(
            "📏 Node Size", 
            min_value=0.5, 
            max_value=2.0, 
            value=1.0, 
            step=0.1,
            help="Adjust the size of graph nodes"
        )    # Simple Manual Edge Management
    with st.expander("🎛️ Customize Graph Display", expanded=False):
        st.markdown("**Control which relationships to show in the graph:**")
        
        # Initialize session state for disabled edges
        if 'disabled_edges' not in st.session_state:
            st.session_state.disabled_edges = set()
        
        # Quick reset option
        if st.button("🔄 Show All Relationships", help="Display all discovered relationships"):
            st.session_state.disabled_edges = set()
            st.rerun()
        
        # Simple edge management
        if st.checkbox("🔧 Select Relationships to Hide", help="Choose which relationships to hide from the graph"):
            # Create simple edge options
            edge_options = []
            for source, target, weight in valid_edges:
                edge_label = f"{display_names[source]} → {display_names[target]}"
                edge_key = f"{source}_{target}"
                edge_options.append((edge_key, edge_label))
            
            if edge_options:
                # Multi-select for disabling edges
                disabled_edge_keys = st.multiselect(
                    "🚫 Select relationships to hide:",
                    options=[key for key, _ in edge_options],
                    format_func=lambda key: next(label for k, label in edge_options if k == key),
                    default=list(st.session_state.disabled_edges),
                    help="Selected relationships will be hidden from the graph visualization"
                )
                
                # Update session state
                st.session_state.disabled_edges = set(disabled_edge_keys)
            else:
                st.info("No relationships available for manual control.")
    
    # Filter edges based on manual selection only
    filtered_edges = []
    for source, target, weight in valid_edges:
        edge_key = f"{source}_{target}"
        if edge_key not in st.session_state.disabled_edges:
            filtered_edges.append((source, target, weight))
    
    if not filtered_edges:
        st.warning("⚠️ No edges available - all relationships have been manually hidden")
        return
    
    # Get selected layout
    pos = layout_options[layout_choice]
    
    # Create the interactive plot
    fig = go.Figure()
      # Add edges with simplified styling
    edge_traces = []
    edge_label_traces = []
    
    for source, target, weight in filtered_edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Simple consistent edge styling
        edge_color = 'rgba(100, 100, 100, 0.7)'  # Gray with transparency
        edge_width = 2  # Consistent width for all edges
        
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
                hoverinfo='text',                hovertext=f"""
                <b>{display_names[source]} → {display_names[target]}</b><br>
                Causal coefficient: {weight:.3f}<br>
                <br>
                <i>This is the estimated direct causal effect size</i>
                """,
                showlegend=False,
                name=f"Edge_{source}_{target}"
            ))
            
            # Enhanced arrow head
            arrow_pos = 0.85
            arrow_x = start_x + arrow_pos * (end_x - start_x)
            arrow_y = start_y + arrow_pos * (end_y - start_y)
            
            arrow_size = 0.04 * node_size_factor  # Consistent arrow size
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
                showlegend=False,                hoverinfo='skip',
                name=f"Arrow_{source}_{target}"
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
        
        node_colors.append(node_color)        # Enhanced hover text
        original_name = G.nodes[node]['original_name']
        display_name = G.nodes[node]['label']
        
        hover_text = f"""
        <b>{display_name}</b><br>
        Role: {'Outcome' if in_degree > out_degree else 'Cause' if out_degree > in_degree else 'Mediator'}<br>
        Incoming relationships: {in_degree}<br>
        Outgoing relationships: {out_degree}<br>
        Network importance: {importance:.3f}<br>
        <br>
        <i>Values shown are DirectLiNGAM causal coefficients</i>
        """
        
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
        ),        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=12, color='black', family='Arial Black'),  # Changed to black for better readability
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False,
        name="Nodes"
    ))
      # Clean layout
    fig.update_layout(
        title=dict(
            text=f"🔗 Interactive Causal Graph ({len(filtered_edges)} relationships)",
            x=0.5,
            font=dict(size=20, color='#2E86AB')
        ),
        showlegend=False,  # No legend needed since all edges are the same style
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),  # Smaller right margin since no legend
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
        }    })    # Simple graph statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔗 Total Relationships", len(filtered_edges))
    
    with col2:
        st.metric("� Variables", len(G.nodes()))
    
    # Edge Status Information
    if st.session_state.get('disabled_edges'):
        total_hidden = len(st.session_state.disabled_edges)
        st.info(f"ℹ️ Currently hiding {total_hidden} relationship(s). Use the 'Customize Graph Display' section above to manage them.")
    
    # Interactive node analysis
    with st.expander("🔍 Detailed Node Analysis", expanded=False):
        selected_node = st.selectbox(
            "Select a variable to analyze:",
            options=display_names,
            help="Choose a variable to see its detailed causal relationships"        )
        
        if selected_node:
            node_idx = display_names.index(selected_node)
            
            # Incoming edges (causes)
            incoming = [(display_names[j], adjacency_matrix[node_idx, j]) for j in range(len(columns)) 
                       if abs(adjacency_matrix[node_idx, j]) > 0.01]
            
            # Outgoing edges (effects)
            outgoing = [(display_names[i], adjacency_matrix[i, node_idx]) for i in range(len(columns))
                       if abs(adjacency_matrix[i, node_idx]) > 0.01]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**📥 What influences {selected_node}:**")
                if incoming:
                    for cause, weight in sorted(incoming, key=lambda x: abs(x[1]), reverse=True):
                        st.write(f"🔗 **{cause}**")
                else:
                    st.write("No direct causes detected")
            
            with col2:
                st.write(f"**📤 What {selected_node} influences:**")
                if outgoing:
                    for effect, weight in sorted(outgoing, key=lambda x: abs(x[1]), reverse=True):
                        st.write(f"🔗 **{effect}**")
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
    # Removed redundant heading - now handled by the main app
    st.write("**Adjust the sliders below to explore different policy scenarios and see the impact in real-time:**")
    
    # Determine which data to use based on variable types
    # If treatment or outcome variables are encoded (end with '_Code'), use encoded data
    use_encoded_data = (treatment_var.endswith('_Code') or outcome_var.endswith('_Code'))
    
    if use_encoded_data and hasattr(analyzer.discovery, 'encoded_data') and analyzer.discovery.encoded_data is not None:
        data_to_use = analyzer.discovery.encoded_data
    else:
        data_to_use = analyzer.data
    
    # Safety check: ensure variables exist in the data
    if treatment_var not in data_to_use.columns:
        st.error(f"❌ Treatment variable '{treatment_var}' not found in data. Available columns: {list(data_to_use.columns)}")
        return
    
    if outcome_var not in data_to_use.columns:
        st.error(f"❌ Outcome variable '{outcome_var}' not found in data. Available columns: {list(data_to_use.columns)}")
        return
    
    # Safety check: ensure variables are numeric
    if not pd.api.types.is_numeric_dtype(data_to_use[treatment_var]):
        st.error(f"❌ Treatment variable '{treatment_var}' is not numeric. Cannot run scenario explorer.")
        return
    
    if not pd.api.types.is_numeric_dtype(data_to_use[outcome_var]):
        st.error(f"❌ Outcome variable '{outcome_var}' is not numeric. Cannot run scenario explorer.")
        return
      # Get current state
    current_treatment_mean = data_to_use[treatment_var].mean()
    current_outcome_mean = data_to_use[outcome_var].mean()
    ate_estimate = ate_results['consensus_estimate']
    
    # Create two columns for scenario controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Scenario 1: Increase Treatment")
        if data_to_use[treatment_var].dtype in ['int64', 'bool']:
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
            delta=f"{scenario1_treatment - current_treatment_mean:+.3f}"        )
        st.metric(
            label=f"{outcome_var} (Scenario 1)", 
            value=f"{scenario1_outcome:.3f}",
            delta=f"{scenario1_change:+.3f}"
        )
    
    with col2:
        st.markdown("#### 📉 Scenario 2: Decrease Treatment")
        if data_to_use[treatment_var].dtype in ['int64', 'bool']:
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
