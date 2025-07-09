"""
CAUSAL AI PLATFORM - MAIN APPLICATION

CRITICAL UNDERSTANDING: This entire script runs from TOP to BOTTOM on:
1. ✅ User opens the page (first visit)
2. ✅ User clicks ANY button 
3. ✅ User uploads a file
4. ✅ User types in text area
5. ✅ User selects from dropdown
6. ✅ User checks a checkbox
7. ✅ ANY user interaction that changes widget state

This file contains the CORE APPLICATION LOGIC:
- 7-Step Causal Analysis Workflow
- Data Processing Pipeline
- User Interactions & State Management
- Main Business Logic

Supporting modules handle:
- UI Styling (ui/styles.py)
- Session Management (utils/session_management.py) 
- Sidebar Components (ui/sidebar.py)
- Data Visualization (ui/components.py)
"""

import streamlit as st
import pandas as pd
import warnings
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Core imports
from causal_ai.analyzer import CausalAnalyzer
from llm.llm import generate_domain_constraints, explain_results_with_llm, suggest_data_requirements, display_data_requirements
from ui.components import (show_data_preview, show_data_quality_summary, 
                          show_correlation_heatmap, show_causal_graph, 
                          show_results_table, show_interactive_scenario_explorer, 
                          show_traditional_comparison)

# New modular imports
from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.workflow_steps import render_step5_causal_inference
from utils.session_management import init_session_state, get_analyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# APPLICATION SETUP & CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Causal AI Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles and render sidebar
apply_custom_styles()
render_sidebar()

# Initialize session state and analyzer
init_session_state()
analyzer = get_analyzer()

# ============================================================================
# HERO SECTION & MAIN APPLICATION INTERFACE  
# ============================================================================

# Hero section with enhanced design
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Causal AI Platform</h1>
    <p class="hero-subtitle">
        Discover <strong>causal relationships</strong> and quantify <strong>treatment effects</strong> with advanced AI-powered analysis.<br>
        Transform your data into actionable insights with cutting-edge algorithms and intelligent domain expertise.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# STEP 0: AI DATA REQUIREMENTS CONSULTANT
# ============================================================================
st.markdown('<div class="step-header"><h2>🤖 Step 0: What Data Do I Need? (AI Consultant)</h2></div>', unsafe_allow_html=True)
st.markdown("*Start here if you're not sure what data to collect for your causal analysis*")

with st.expander("💡 Not sure what data you need? Let AI help you!", expanded=False):
    st.markdown("""
    **Perfect for:**
    - Planning a new data collection project
    - Understanding what variables are important for causal analysis
    - Getting guidance on research design
    
    **How it works:**
    1. Describe your business problem
    2. AI suggests what data to collect
    3. Get practical collection methods and tips
    """)
    
    # Business problem input
    business_problem = st.text_area(
        "Describe your business problem:",
        placeholder="e.g., 'We want to understand if our marketing campaigns actually increase sales, or if it's just correlation. We need to optimize our marketing budget allocation.'",
        height=100,
        key="business_problem_input",
        help="Be specific about what you want to achieve and what decisions you need to make"
    )
    
    # Button to get AI suggestions
    col1, col2 = st.columns([1, 2])
    with col1:
        get_requirements_btn = st.button(
            "🧠 Get Data Requirements", 
            type="primary",
            key="get_requirements_btn",
            disabled=not business_problem.strip(),
            help="AI will suggest what data to collect for your problem"
        )
    
    with col2:
        if not st.session_state.get('openai_api_key'):
            st.warning("🔑 Add OpenAI API key in sidebar to use AI consultant")
    
    # Get AI suggestions
    if get_requirements_btn and business_problem.strip():
        if st.session_state.get('openai_api_key'):
            with st.spinner("🤖 AI is analyzing your business problem..."):
                requirements = suggest_data_requirements(
                    business_problem.strip(),
                    st.session_state.get('openai_api_key')
                )
                
                # Store results in session state for persistence
                st.session_state['data_requirements'] = requirements
                st.session_state['requirements_generated'] = True
                
        else:
            st.error("Please add your OpenAI API key in the sidebar to use the AI consultant")
    
    # Display results if available
    if st.session_state.get('requirements_generated') and st.session_state.get('data_requirements'):
        st.markdown("### 📊 **AI Data Requirements for Your Problem**")
        display_data_requirements(st.session_state['data_requirements'])
        
        # Option to clear requirements
        if st.button("🗑️ Clear Requirements", key="clear_requirements"):
            st.session_state['data_requirements'] = None
            st.session_state['requirements_generated'] = False
            st.rerun()
        
        # Transition message
        st.success("✅ **Next Step**: Once you've collected this data, upload it below to start your causal analysis!")

# ============================================================================
# STEP 1: DATA UPLOAD & MANAGEMENT
# ============================================================================
step1_class = "step-completed" if st.session_state.get('data_loaded', False) else ""
st.markdown(f'<div class="step-header {step1_class}"><h2>📁 Step 1: Data Upload</h2></div>', unsafe_allow_html=True)

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
    sample_data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    sample_files = []
    
    # Import sample dataset configurations
    from config.sample_data import SAMPLE_DATASETS
    sample_descriptions = {filename: config['display_name'] for filename, config in SAMPLE_DATASETS.items()}
    
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
                help="Sample dataset demonstrates causal relationships in supply chain logistics and environmental impact analysis.",                key="sample_dataset_selector"
            )
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            load_button_disabled = selected_sample == 'None'
            
            if st.button("🚀 Load Dataset", key="load_sample", disabled=load_button_disabled, use_container_width=True):
                sample_file_path = os.path.join(sample_data_dir, selected_sample)
                try:
                    # Load the sample data
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
    
    # ============================================================================
    # STEP 2: DOMAIN CONSTRAINTS (AI-POWERED)
    # ============================================================================
    step2_class = "step-completed" if st.session_state.get('domain_constraints_generated', False) else ""
    st.markdown(f'<div class="step-header {step2_class}"><h2>🧠 Step 2: Domain Constraints (AI-Powered)</h2></div>', unsafe_allow_html=True)
    
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
    
    # ============================================================================
    # STEP 3: CAUSAL DISCOVERY
    # ============================================================================
    step3_completed = st.session_state.get('causal_discovery_completed', False)
    step3_class = "step-completed" if step3_completed else ""
    st.markdown(f'<div class="step-header {step3_class}"><h2>🔍 Step 3: Causal Discovery</h2></div>', unsafe_allow_html=True)
    
    # Add validation for constraints
    constraints_ready = (
        st.session_state.get('domain_constraints_generated') or 
        st.checkbox("Skip AI constraints (use default)", help="Run discovery without AI-generated constraints", key="skip_constraints_checkbox")    )
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
        
        show_causal_graph(analyzer.adjacency_matrix, graph_columns)    
    
    # ============================================================================
    # STEP 4: VARIABLE RELATIONSHIP ANALYSIS
    # ============================================================================
    st.markdown('<div class="step-header"><h2>📊 Step 4: Variable Relationship Analysis</h2></div>', unsafe_allow_html=True)
    
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
    # ============================================================================
    # STEP 5: CAUSAL INFERENCE ANALYSIS
    # ============================================================================
    render_step5_causal_inference(analyzer)
    
    # Display results if available
    if st.session_state.get('ate_results'):
        ate_results = st.session_state['ate_results']
        treatment_var = st.session_state['selected_treatment']
        outcome_var = st.session_state['selected_outcome']
        
        # Main result
        st.subheader("📊 Main Result")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Add effect size context to the metric using config constants
            from causal_ai.config import EFFECT_SIZE_THRESHOLDS
            
            effect_size = abs(ate_results.get('consensus_estimate', 0))
            if effect_size > EFFECT_SIZE_THRESHOLDS['large']:
                effect_label = "Large Effect"
            elif effect_size > EFFECT_SIZE_THRESHOLDS['moderate']:
                effect_label = "Moderate Effect"
            elif effect_size > EFFECT_SIZE_THRESHOLDS['small']:
                effect_label = "Small Effect"
            else:
                effect_label = "Very Small Effect"
                
            st.metric(
                f"Causal Effect Estimate ({effect_label})", 
                f"{ate_results['consensus_estimate']:.4f}"
            )
        with col2:
            st.info(f"**Interpretation:** {ate_results['interpretation']}")
        
        # Detailed results
        st.subheader("🔍 Detailed Results by Method")
        show_results_table(ate_results)
        
        # Additional Information: Compare with Traditional Methods
        st.markdown("### 🔬 Compare with Traditional Statistical Methods")
        st.markdown("*Optional: See how causal AI results compare to traditional approaches*")
        
        # Show comparison button and results in a persistent container
        comparison_container = st.container()
        with comparison_container:
            if not st.session_state.get('traditional_results'):
                if st.button("🔄 Run Traditional Analysis Comparison", type="secondary", key="run_traditional_comparison"):
                    with st.spinner("Running traditional statistical analysis..."):
                        from utils.traditional_analysis import run_traditional_analysis, compare_causal_vs_traditional
                        
                        # Run traditional analysis
                        traditional_results = run_traditional_analysis(analyzer, treatment_var, outcome_var)
                        
                        # Compare results
                        comparison = compare_causal_vs_traditional(ate_results, traditional_results, treatment_var, outcome_var)
                          # Store results in session state
                        st.session_state['traditional_results'] = traditional_results
                        st.session_state['comparison_results'] = comparison
                        st.session_state['last_action'] = 'traditional_analysis_completed'
                        st.rerun()  # Refresh to show results
            
            # Always display results if available (persistent across all interactions)
            if st.session_state.get('traditional_results') and st.session_state.get('comparison_results'):
                st.success("✅ Traditional analysis comparison completed!")
                
                trad_results = st.session_state['traditional_results']
                comp_results = st.session_state['comparison_results']
                
                # Simple comparison table
                causal_estimate = ate_results.get('consensus_estimate', 0)
                
                st.markdown("**Method Comparison:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🧠 Causal AI", f"{causal_estimate:.4f}")
                with col2:
                    if 'correlation' in trad_results['methods']:
                        corr_est = trad_results['methods']['correlation'].get('estimate', 0)
                        st.metric("📈 Correlation", f"{corr_est:.4f}")
                with col3:
                    if 'simple_regression' in trad_results['methods']:
                        reg_est = trad_results['methods']['simple_regression'].get('estimate', 0)
                        st.metric("📊 Simple Regression", f"{reg_est:.4f}")
                with col4:
                    if st.button("🗑️ Clear Comparison", key="clear_comparison"):
                        st.session_state['traditional_results'] = None
                        st.session_state['comparison_results'] = None
                        st.rerun()
                
                # Single key insight
                if comp_results['key_differences']:
                    max_diff = max([diff['percent_difference'] for diff in comp_results['key_differences']])
                    if max_diff > 20:
                        st.info(f"💡 **Key Insight**: Causal AI differs by up to {max_diff:.0f}% from traditional methods, accounting for confounding factors that traditional methods miss.")
                    else:
                        st.success("✅ **Consistent results** across methods")
                else:
                    st.success("✅ **Consistent results** across methods")
          
          # ============================================================================
          # STEP 6: INTERACTIVE POLICY EXPLORER
          # ============================================================================
        if abs(ate_results['consensus_estimate']) > 0.01:  # Only show if we have a meaningful effect
            st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', unsafe_allow_html=True)
            st.info("💡 **Explore Policy Scenarios:** Use the estimated causal effect to simulate different intervention strategies.")
            show_interactive_scenario_explorer(ate_results, treatment_var, outcome_var, analyzer)
        else:
            st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', unsafe_allow_html=True)
            st.warning("⚠️ **Policy Explorer unavailable:** The estimated causal effect is too small (≈0) to provide meaningful scenario predictions.")
          
          # ============================================================================
          # STEP 7: AI-POWERED INSIGHTS
          # ============================================================================
        st.markdown('<div class="step-header"><h2>🧠 Step 7: AI-Powered Insights</h2></div>', unsafe_allow_html=True)
        
        # Show button only if API key is available
        if st.session_state.get('openai_api_key'):
            if st.button("🤖 Get Detailed AI Analysis", type="secondary", key="get_ai_analysis_btn"):
                with st.spinner("AI is analyzing your results..."):
                    explanation = explain_results_with_llm(
                        ate_results, treatment_var, outcome_var,
                        st.session_state.get('openai_api_key')
                    )
                
                st.markdown("### 📋 Business Insights & Recommendations")
                st.markdown(explanation)
        else:
            st.warning("🔑 Add OpenAI API key in sidebar to get AI-powered explanations")

# ============================================================================
# LANDING PAGE (NO DATA LOADED)
# ============================================================================
else:
    # Landing page when no file is uploaded
    st.info("""
    👆 **Get Started:** Upload your Excel or CSV file above to begin causal analysis.
    
    **What you can do with this platform:**
    - 🔍 Discover causal relationships in your data
    - 📊 Calculate treatment effects with confidence intervals
    - 🤖 Get AI-powered insights and recommendations
    - 🎯 Simulate policy interventions
    
    **Try our sample dataset:** 🌱 CO2 Supply Chain Analysis - explore how transportation factors affect environmental impact!
    """)
