"""
Workflow Step Components for Causal AI Platform
Modular step implementations to reduce main file complexity
"""

import streamlit as st
from ui.components import show_data_preview, show_data_quality_summary

def render_step1_data_upload(analyzer):
    """Step 1: Data Upload and Management"""
    step1_class = "step-completed" if st.session_state.get('data_loaded', False) else ""
    st.markdown(f'<div class="step-header {step1_class}"><h2>📁 Step 1: Data Upload</h2></div>', 
                unsafe_allow_html=True)
    
    # Create tabs for better organization - maintain active tab state
    tab1, tab2 = st.tabs(["📤 Upload Your Own Data", "📊 Use Sample Dataset"])

    # Show active data source info above tabs if sample data is loaded
    if st.session_state.get('sample_data_loaded'):
        # More prominent display for recently loaded data
        if st.session_state.get('last_action') == 'sample_data_loaded':
            st.success(f"🎉 **Dataset Ready**: {st.session_state['sample_data_loaded']} is now loaded and ready for analysis!")
        else:
            st.success(f"✅ **Active Dataset**: {st.session_state['sample_data_loaded']} (loaded from sample data)")

    with tab1:
        st.markdown("Upload your Excel or CSV file to begin causal analysis:")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['xlsx', 'csv'],
            help="Ensure your data has clean column names and numeric values for best results.",
            key="file_uploader_main"
        )
        
        if uploaded_file:
            st.info(f"📄 **File selected:** {uploaded_file.name}")

    with tab2:
        st.markdown("Explore causal relationships with our pre-loaded demonstration datasets:")
        
        # Get list of sample data files
        import os
        sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_data')
        sample_files = []
        sample_descriptions = {
            'CO2 SupplyChain Demo.csv': '🌱 Supply Chain CO2 Emissions (Logistics Impact Analysis)'
        }
        
        if os.path.exists(sample_data_dir):
            for file in os.listdir(sample_data_dir):
                if file.endswith('.csv'):
                    sample_files.append(file)
        
        if sample_files:
            # Create columns for better layout
            col_select, col_button = st.columns([2, 1])
            
            with col_select:
                selected_sample = st.selectbox(
                    "Choose a sample dataset:",
                    options=['None'] + sample_files,
                    format_func=lambda x: 'Select a dataset...' if x == 'None' else sample_descriptions.get(x, x),
                    help="Sample dataset demonstrates causal relationships in supply chain logistics and environmental impact analysis.",
                    key="sample_dataset_selector"
                )        
            with col_button:
                st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                load_button_disabled = selected_sample == 'None'
                
                if st.button("🚀 Load Dataset", key="load_sample", disabled=load_button_disabled, use_container_width=True):
                    sample_file_path = os.path.join(sample_data_dir, selected_sample)
                    try:
                        # Load the sample data
                        import pandas as pd
                        sample_data = pd.read_csv(sample_file_path)
                        analyzer.data = sample_data
                        
                        # Set session state
                        st.session_state['data_loaded'] = True
                        st.session_state['sample_data_loaded'] = selected_sample
                        st.session_state['last_action'] = 'sample_data_loaded'
                        
                        # Trigger rerun to show the persistent success message above
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading sample dataset: {str(e)}")
                        
        else:
            st.info("No sample datasets found. Run `python create_sample_data.py` to generate sample datasets.")

    # Handle uploaded file
    data_source = None
    if uploaded_file:
        if analyzer.load_data(uploaded_file):
            # 🚦 STATE TRANSITION: Data uploaded
            st.session_state['data_loaded'] = True
            st.session_state['causal_discovery_completed'] = False
            st.session_state.pop('sample_data_loaded', None)  # Clear sample data flag
            data_source = f"Uploaded file: {uploaded_file.name}"
            
            st.success("✅ Data loaded successfully!")
    elif st.session_state.get('sample_data_loaded'):
        data_source = f"Sample dataset: {st.session_state['sample_data_loaded']}"

    # Show data info if data is loaded
    if st.session_state.get('data_loaded') and analyzer.data is not None:
        
        # Display data preview and quality summary 
        show_data_preview(analyzer.data)
        show_data_quality_summary(analyzer.data)
        
        # Add sample dataset information if applicable
        if st.session_state.get('sample_data_loaded'):
            sample_name = st.session_state['sample_data_loaded']
            
            # Show sample dataset description
            if sample_name == 'CO2 SupplyChain Demo.csv':
                st.info("""
                🌱 **Supply Chain CO2 Emissions Analysis**: This dataset examines factors affecting CO2 emissions in agricultural supply chains.
                Variables include transportation method, distance, fuel type, vehicle type, weather conditions, and resulting emissions. 
                The goal is to identify causal factors that could reduce environmental impact in logistics operations.
                """)
    
    return uploaded_file

def render_step2_domain_constraints(analyzer):
    """Step 2: Domain Knowledge & Constraints"""
    if not st.session_state.get('data_loaded'):
        st.warning("⚠️ Please upload data first")
        return
    
    step2_class = "step-completed" if st.session_state.get('domain_constraints_generated', False) else ""
    st.markdown(f'<div class="step-header {step2_class}"><h2>🧠 Step 2: Domain Knowledge & Constraints</h2></div>', 
                unsafe_allow_html=True)
    
    domain_context = st.text_area(
        "Describe your domain/business context:",
        placeholder="e.g., 'Distance causes CO2 emissions; Weather cannot affect fuel consumption'",
        height=100,
        key="domain_context_input",
        value=st.session_state.get('domain_context_text', '')
    )
    
    # Show help for writing domain constraints
    with st.expander("💡 How to Write Good Domain Constraints", expanded=False):
        from llm.llm import get_domain_constraints_help
        st.markdown(get_domain_constraints_help())
    
    # Store the domain context in session state
    if domain_context:
        st.session_state['domain_context_text'] = domain_context
    
    if st.button("Generate AI Constraints", type="secondary", key="generate_constraints_btn"):
        if domain_context.strip():
            # Clear any existing constraints to start fresh
            st.session_state['constraints_generated'] = False
            st.session_state['domain_constraints_generated'] = False
            st.session_state['constraints_data'] = None
            
            with st.spinner("Analyzing domain context..."):
                try:
                    from llm.llm import generate_domain_constraints
                    suggested_constraints = generate_domain_constraints(
                        list(analyzer.data.columns), 
                        domain_context,
                        st.session_state.get('openai_api_key')
                    )
                    
                    if suggested_constraints and (suggested_constraints.get('forbidden_edges') or suggested_constraints.get('required_edges')):
                        st.session_state['suggested_constraints'] = suggested_constraints
                        st.session_state['constraints_generated'] = True
                    else:
                        st.warning("No specific constraints extracted from context")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Domain context required")
    
    # Show simple approval interface if constraints were generated
    if st.session_state.get('constraints_generated') and st.session_state.get('suggested_constraints'):
        from llm.llm import display_simple_constraint_approval
        
        st.markdown("### 🤖 **AI-Generated Constraints**")
        # Get user approval/rejection
        approved_constraints = display_simple_constraint_approval(st.session_state['suggested_constraints'])
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Add Selected to Pool", type="primary", key="add_ai_constraints"):
                # Add approved constraints to the constraint pool
                if st.session_state.get('constraints_data'):
                    # Merge with existing constraints
                    existing = st.session_state['constraints_data']
                    from llm.llm import combine_ai_and_manual_constraints
                    combined = combine_ai_and_manual_constraints(existing, approved_constraints)
                    st.session_state['constraints_data'] = combined
                else:
                    # First constraints being added
                    st.session_state['constraints_data'] = approved_constraints
                
                st.session_state['domain_constraints_generated'] = True
                st.session_state['constraints_generated'] = False
                st.success(f"✅ Added {len(approved_constraints.get('required_edges', [])) + len(approved_constraints.get('forbidden_edges', []))} AI constraints to pool")
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip AI Constraints", key="skip_ai_constraints"):
                st.session_state['constraints_generated'] = False
                st.rerun()
    
    # Manual constraint builder
    with st.expander("➕ Add Manual Constraints", expanded=False):
        from llm.llm import display_manual_constraint_builder
        
        st.markdown("**Add your own domain knowledge:**")
        display_manual_constraint_builder(list(analyzer.data.columns))
    
    # Show unified constraint review section
    if st.session_state.get('domain_constraints_generated') and st.session_state.get('constraints_data'):
        constraints_data = st.session_state['constraints_data']
        
        st.markdown("### 📋 **Review All Constraints**")
        st.markdown("*Remove any constraints you don't want to apply:*")
        
        # Create a modifiable copy for editing
        modified_constraints = {
            'required_edges': constraints_data.get('required_edges', []).copy(),
            'forbidden_edges': constraints_data.get('forbidden_edges', []).copy(),
            'explanation': constraints_data.get('explanation', '')
        }
        
        constraints_modified = False
        
        # Show required edges with remove buttons
        if modified_constraints['required_edges']:
            st.markdown("**✅ Required Relationships (A must cause B):**")
            for i, edge in enumerate(modified_constraints['required_edges']):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"• **{edge[0]}** → **{edge[1]}**")
                with col2:
                    if st.button("🗑️", key=f"remove_required_{i}", help="Remove this constraint"):
                        modified_constraints['required_edges'].remove(edge)
                        constraints_modified = True
        
        # Show forbidden edges with remove buttons  
        if modified_constraints['forbidden_edges']:
            st.markdown("**🚫 Forbidden Relationships (A cannot cause B):**")
            for i, edge in enumerate(modified_constraints['forbidden_edges']):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"• **{edge[0]}** ✗ **{edge[1]}**")
                with col2:
                    if st.button("🗑️", key=f"remove_forbidden_{i}", help="Remove this constraint"):
                        modified_constraints['forbidden_edges'].remove(edge)
                        constraints_modified = True
        
        # Update session state if constraints were modified
        if constraints_modified:
            st.session_state['constraints_data'] = modified_constraints
            st.rerun()
            
        # Action buttons
        total_constraints = len(modified_constraints['required_edges']) + len(modified_constraints['forbidden_edges'])
        
        if total_constraints > 0:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Apply These Constraints", type="primary", key="apply_final_constraints"):
                    analyzer.domain_constraints = modified_constraints
                    st.success(f"🎯 Applied {total_constraints} constraints for causal discovery!")
                    
            with col2:
                if st.button("🗑️ Clear All Constraints", key="clear_all_constraints"):
                    st.session_state['constraints_data'] = None
                    st.session_state['domain_constraints_generated'] = False
                    st.session_state['constraints_generated'] = False
                    analyzer.domain_constraints = None
                    st.rerun()
        else:
            st.info("No constraints in pool. Add some above or skip to use defaults.")

def render_step3_causal_discovery(analyzer):
    """Step 3: Causal Discovery"""
    if not st.session_state.get('data_loaded'):
        st.warning("⚠️ Please upload data first")
        return
    
    step3_completed = st.session_state.get('causal_discovery_completed', False)
    step3_class = "step-completed" if step3_completed else ""
    st.markdown(f'<div class="step-header {step3_class}"><h2>🔍 Step 3: Causal Discovery</h2></div>', 
                unsafe_allow_html=True)
    
    # Add validation for constraints
    constraints_ready = (
        st.session_state.get('domain_constraints_generated') or 
        st.checkbox("Skip AI constraints (use default)", help="Run discovery without AI-generated constraints", key="skip_constraints_checkbox")
    )
    
    if st.button("🚀 Run Causal Discovery", type="primary", disabled=not constraints_ready, key="run_causal_discovery_btn"):
        with st.spinner("Discovering causal relationships..."):
            active_constraints = st.session_state.get('constraints_data', {})
            success = analyzer.run_causal_discovery(active_constraints)
        
        if success:
            st.session_state['causal_discovery_completed'] = True
            st.session_state['last_action'] = 'discovery_completed'
            
            # Success message
            st.success("🎉 Causal discovery completed successfully!")
            
            if active_constraints:
                constraint_count = len(active_constraints.get('forbidden_edges', [])) + len(active_constraints.get('required_edges', []))
                st.info(f"🧠 Applied {constraint_count} AI-powered domain constraints to guide discovery")
            
            # Trigger page refresh to show updated step header
            st.rerun()
        else:
            st.session_state['causal_discovery_completed'] = False
            st.session_state['last_action'] = 'discovery_failed'
            
            # Error message
            st.error("❌ Causal discovery encountered an error")
            
            # Show helpful guidance for next steps
            with st.expander("💡 Troubleshooting Help", expanded=False):
                st.markdown("""
                **Common issues and solutions:**
                - **Missing dependencies**: Install required packages with `pip install lingam`
                - **Insufficient data**: Ensure your dataset has at least 2 numeric columns
                - **Data quality**: Check for missing values, constant columns, or data formatting issues
                - **Constraints conflicts**: Review your domain constraints for logical inconsistencies
                
                **Next steps:**
                1. Review the error message above for specific details
                2. Check your data quality in Step 1
                3. Simplify or remove domain constraints in Step 2
                4. Try with a different dataset or data subset
                """)
    
    # Show causal discovery results
    if st.session_state['causal_discovery_completed'] and analyzer.adjacency_matrix is not None:
        st.subheader("📈 Discovered Causal Graph")
        
        # Use columns from discovery (these match the adjacency matrix dimensions)
        if analyzer.columns:
            graph_columns = analyzer.columns
            st.info(f"📊 Showing relationships between {len(graph_columns)} variables")
            
            # Add simplified explanation
            st.markdown("""
            **How to read this graph:**
            - 🔵 **Nodes** = Variables in your dataset
            - ➡️ **Arrows** = Causal relationships (A → B means A causes B)
            - 🎨 **Node Colors** = Variable role (Red=Outcome, Teal=Cause, Blue=Mediator)
            - 🎛️ **Customize** = Use the controls below to adjust layout and hide relationships""")
        else:
            # Fallback to original columns if columns not available
            graph_columns = list(analyzer.data.columns)
        
        from ui.components import show_causal_graph
        show_causal_graph(analyzer.adjacency_matrix, graph_columns)

def render_step4_relationship_analysis(analyzer):
    """Step 4: Variable Relationship Analysis"""
    st.markdown('<div class="step-header"><h2>📊 Step 4: Variable Relationship Analysis</h2></div>', 
                unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded'):
        st.warning("⚠️ Please upload data first")
        return
    
    # Container for persistent results
    step4_container = st.container()
    
    with step4_container:
        if st.button("🔍 Analyze Variable Relationships", type="secondary", key="analyze_relationships_btn"):
            with st.spinner("Analyzing variable relationships..."):
                relationships = analyzer.analyze_variable_relationships()
                
                if relationships:
                    # Store in session state for persistence
                    st.session_state['step4_relationships'] = relationships
                    st.session_state['step4_completed'] = True
                    st.session_state['last_action'] = 'relationships_analyzed'
        
        # Display results if available (persistent across UI updates)
        if st.session_state.get('step4_completed') and st.session_state.get('step4_relationships'):
            relationships = st.session_state['step4_relationships']
            st.success("✅ Relationship analysis completed!")
            
            # Show info about numeric vs categorical variables
            if 'categorical_columns' in relationships and relationships['categorical_columns']:
                st.info(f"📊 Correlation analysis performed on {len(relationships['numeric_columns'])} numeric variables. {len(relationships['categorical_columns'])} categorical variables were excluded.")
                with st.expander("ℹ️ Excluded Categorical Variables"):
                    st.write(f"Categorical variables excluded from correlation analysis: {', '.join(relationships['categorical_columns'])}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🔥 Correlation Heatmap (Numeric Variables)")
                if not relationships['correlation_matrix'].empty:
                    from ui.components import show_correlation_heatmap
                    show_correlation_heatmap(relationships['correlation_matrix'])
                else:
                    st.warning("No numeric data available for correlation analysis")
            
            with col2:
                st.subheader("💪 Strongest Correlations")
                strong_corr = relationships['strong_correlations']
                if strong_corr:
                    st.write(f"Found {len(strong_corr)} strong correlations (|r| > 0.5):")
                    for corr in strong_corr[:5]:
                        st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
                else:
                    st.write("No strong correlations (>0.5) found")

def render_step5_causal_inference(analyzer):
    """Step 5: Causal Inference Analysis"""
    st.markdown('<div class="step-header"><h2>🔬 Step 5: Causal Inference Analysis</h2></div>', 
                unsafe_allow_html=True)
    
    # Check if causal discovery has been run
    if analyzer.adjacency_matrix is None:
        st.warning("⚠️ **Causal discovery must be run before causal inference.** Please complete Step 3 first.")
        st.info("💡 **Why this matters:** Causal inference requires understanding the causal structure between variables, which is discovered in Step 3.")
        return
    elif analyzer.data is not None and not analyzer.data.empty:        # Get columns for causal analysis
        numeric_columns = analyzer.get_numeric_columns()
        categorical_columns = analyzer.get_categorical_columns()
        all_columns = numeric_columns + categorical_columns
        
        # Show comprehensive variable information (consolidated message)
        if categorical_columns:
            st.info(f"""
            **Variable Classification for Causal Analysis:**
            
            • **Numeric variables** ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None found'}
            • **Categorical variables** ({len(categorical_columns)}): {', '.join(categorical_columns)}
            
            ℹ️ **Note:** Both numeric and categorical treatments are supported. Categorical treatments use specialized policy scenarios.
            """)
        else:
            st.info(f"📊 **Available Variables:** {len(numeric_columns)} numeric variables: {', '.join(numeric_columns) if numeric_columns else 'None found'}")
        
        if len(all_columns) < 2:
            st.error("❌ Need at least 2 variables for causal analysis. Please ensure your data contains numeric or categorical columns.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            treatment_var = st.selectbox(
                "Treatment Variable (Cause)",
                options=all_columns,
                key="treatment_select",
                help="Select a variable that represents the intervention or treatment (both numeric and categorical treatments are supported)"
            )
        
        with col2:
            # Outcome should be numeric for meaningful measurement
            available_outcomes = [col for col in numeric_columns if col != treatment_var]
            if not available_outcomes:
                st.error(f"❌ **No numeric outcome variables available.** Selected treatment: '{treatment_var}' ({'numeric' if treatment_var in numeric_columns else 'categorical'})")
                st.info("💡 **Solution:** Select a different treatment variable or ensure your data has numeric outcome variables.")
                return
            
            # Preserve the previous outcome selection if it's still valid
            previous_outcome = st.session_state.get('previous_outcome_var')
            default_index = 0
            
            if previous_outcome and previous_outcome in available_outcomes:
                try:
                    default_index = available_outcomes.index(previous_outcome)
                except ValueError:
                    default_index = 0
                
            outcome_var = st.selectbox(
                "Outcome Variable (Effect)", 
                options=available_outcomes,
                index=default_index,
                key="outcome_select",
                help="Select a numeric variable that represents the outcome you want to measure"            )
            
            # Store the current outcome selection for next time
            st.session_state['previous_outcome_var'] = outcome_var
        
        if st.button("🔬 Run Causal Inference", type="primary", key="run_causal_inference_btn"):
            if treatment_var != outcome_var:
                if outcome_var not in numeric_columns:
                    st.error(f"❌ **Outcome variable '{outcome_var}' must be numeric.**")
                    return
                
                # Debug mode toggle for categorical inference
                if treatment_var in categorical_columns:
                    debug_mode = st.checkbox("🔍 Enable debug mode for categorical treatment", 
                                           help="Show detailed information about categorical variable handling")
                else:
                    debug_mode = False
                
                with st.spinner("Running causal inference..."):
                    try:
                        # Debug output for categorical treatments
                        if debug_mode and treatment_var in categorical_columns:
                            st.info("🔍 **Debug Mode: Categorical Treatment Analysis**")
                            
                            # Show categorical mappings if available
                            if analyzer.categorical_mappings and treatment_var in analyzer.categorical_mappings:
                                mapping = analyzer.categorical_mappings[treatment_var]
                                st.write(f"**Categorical mappings for '{treatment_var}':**")
                                for original, encoded in mapping['mapping'].items():
                                    st.write(f"  • {original} → {encoded}")
                            
                            # Show data source information
                            st.write(f"**Data handling:**")
                            st.write(f"  • Treatment '{treatment_var}' is categorical")
                            st.write(f"  • Outcome '{outcome_var}' is numeric")
                            
                            if analyzer.encoded_data is not None:
                                if treatment_var in analyzer.encoded_data.columns:
                                    st.write(f"  • Using encoded data for categorical treatment")
                                    st.write(f"  • Encoded data shape: {analyzer.encoded_data.shape}")
                                else:
                                    st.write(f"  • Treatment not found in encoded data - this may cause issues")
                            else:
                                st.write(f"  • No encoded data available - this may cause issues for categorical treatments")
                        
                        ate_results = analyzer.calculate_ate(treatment_var, outcome_var)
                        st.session_state['ate_results'] = ate_results
                        st.session_state['selected_treatment'] = treatment_var
                        st.session_state['selected_outcome'] = outcome_var
                        st.session_state['last_action'] = 'inference_completed'
                        
                        # Simple success message
                        st.success("✅ Causal inference completed successfully!")
                        
                    except Exception as e:
                        # Error message with actual error details
                        st.error(f"❌ **Causal Inference Error:** {str(e)}")
                        st.info("💡 **Troubleshooting:** Try selecting different variables or check data quality.")
            else:
                st.error("❌ Please select different variables for treatment and outcome")

def render_step6_policy_explorer(analyzer):
    """Step 6: Interactive Policy Explorer"""
    if not st.session_state.get('ate_results'):
        st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', 
                    unsafe_allow_html=True)
        st.info("💡 Complete causal inference analysis first to explore policy scenarios")
        return
    
    ate_results = st.session_state['ate_results']
    treatment_var = st.session_state['selected_treatment']
    outcome_var = st.session_state['selected_outcome']
    
    if abs(ate_results['consensus_estimate']) > 0.01:  # Only show if we have a meaningful effect
        st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', 
                    unsafe_allow_html=True)
        st.info("💡 **Explore Policy Scenarios:** Use the estimated causal effect to simulate different intervention strategies.")
        from ui.components import show_interactive_scenario_explorer
        show_interactive_scenario_explorer(ate_results, treatment_var, outcome_var, analyzer)
    else:
        st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', 
                    unsafe_allow_html=True)
        st.warning("⚠️ **Policy Explorer unavailable:** The estimated causal effect is too small (≈0) to provide meaningful scenario predictions.")

def render_step7_ai_insights(analyzer):
    """Step 7: AI-Powered Insights"""
    st.markdown('<div class="step-header"><h2>🧠 Step 7: AI-Powered Insights</h2></div>', 
                unsafe_allow_html=True)
    
    if not st.session_state.get('ate_results'):
        st.info("💡 Complete causal inference analysis first to get AI-powered insights")
        return
    
    ate_results = st.session_state['ate_results']
    treatment_var = st.session_state['selected_treatment']
    outcome_var = st.session_state['selected_outcome']
    
    # Show button only if API key is available
    if st.session_state.get('openai_api_key'):
        if st.button("🤖 Get Detailed AI Analysis", type="secondary", key="get_ai_analysis_btn"):
            with st.spinner("AI is analyzing your results..."):
                from llm.llm import explain_results_with_llm
                explanation = explain_results_with_llm(
                    ate_results, treatment_var, outcome_var,
                    st.session_state.get('openai_api_key')
                )
            
            st.markdown("### 📋 Business Insights & Recommendations")
            st.markdown(explanation)
    else:
        st.warning("🔑 Add OpenAI API key in sidebar to get AI-powered explanations")
