# CRITICAL UNDERSTANDING: This entire script runs from TOP to BOTTOM on:
# 1. ✅ User opens the page (first visit)
# 2. ✅ User clicks ANY button 
# 3.    st.markdown("### 🔬 **Core Methodology**")✅ User uploads a file
# 4. ✅ User types in text area
# 5. ✅ User selects from dropdown
# 6. ✅ User checks a checkbox
# 7. ✅ ANY user interaction that changes widget state

import streamlit as st
import pandas as pd
import warnings
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from causal_ai.analyzer import CausalAnalyzer
from llm.llm import generate_domain_constraints, explain_results_with_llm
from ui.components import show_data_preview, show_data_quality_summary, show_correlation_heatmap, show_causal_graph, show_results_table, show_interactive_scenario_explorer, show_traditional_comparison

# Comprehensive warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure page
st.set_page_config(
    page_title="Causal AI Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }    .hero-section {
        text-align: center; 
        padding: 2.5rem 2rem; 
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
        border-radius: 15px; 
        margin-bottom: 2.5rem; 
        color: #2c3e50;
        box-shadow: 0 10px 30px rgba(255, 154, 158, 0.3);
    }    .hero-title {
        font-size: 2.8rem; 
        font-weight: 700; 
        margin-bottom: 0.8rem; 
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        letter-spacing: -0.02em;        color: #2c3e50;
    }    .hero-subtitle {
        font-size: 1.2rem; 
        font-weight: 400; 
        margin: 1rem 0 0 0; 
        opacity: 0.95; 
        line-height: 1.7;
        max-width: 95%;
        margin-left: auto;
        margin-right: auto;
        color: #2c3e50;
        text-align: center;
        padding: 0 1rem;
    }
    .step-header {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #e17055;
        border-radius: 0 10px 10px 0;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
    }
    .step-header h2 {
        margin: 0;
        color: #2c3e50;
        font-weight: 600;
    }
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }    .stButton > button {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2c3e50;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 154, 158, 0.4);
    }    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #fff5f5 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    # This creates ONE shared analyzer instance for ALL users
    # But each user's data and results are kept separate via session_state
    return CausalAnalyzer()

analyzer = get_analyzer()

# Initialize session state with default values
def init_session_state():
    """Initialize session state variables with defaults"""
    defaults = {
        'causal_discovery_completed': False,
        'data_loaded': False,
        'selected_treatment': None,
        'selected_outcome': None,
        'ate_results': None,
        'domain_constraints_generated': False,
        'constraints_data': None,
        'openai_api_key': '',
        'api_key_valid': False,
        'last_verified_key': '',
        'active_data_tab': 0,  # 0 = Upload tab, 1 = Sample tab
        'previous_outcome_var': None,  # Store previous outcome selection
        'domain_context_text': '',  # Persist domain context
        'step4_completed': False,
        'step4_relationships': None,
        'traditional_results': None,
        'comparison_results': None,
        'constraints_display': None,
        'data_quality_expanded': False,  # Expander states
        'step4_expander_state': False,
        'keep_scroll_position': False,  # Prevent page jumps
        'last_action': None,  # Track last user action
        # Simple constraint approval variables
        'suggested_constraints': None,  # Store AI suggestions
        'constraints_generated': False,  # Flag for showing approval interface
        'manual_constraints': {"forbidden_edges": [], "required_edges": []}  # Manual constraints
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
      # API Key input with automatic verification
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Required for AI-powered constraint generation",
        value=st.session_state.get('openai_api_key', ''),
        placeholder="sk-proj-..."
    )
    
    # Auto-verify API key when it changes
    if api_key and api_key != st.session_state.get('last_verified_key', ''):
        with st.spinner("Verifying API key..."):
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Quick test call to verify the key works
                response = client.models.list()
                st.session_state['api_key_valid'] = True
                st.session_state['last_verified_key'] = api_key
            except Exception as e:
                st.session_state['api_key_valid'] = False
                st.session_state['last_verified_key'] = api_key
    elif not api_key:
        st.session_state['api_key_valid'] = False
        st.session_state['last_verified_key'] = ''
    
    # Save API key to session state and show status
    if api_key:
        st.session_state['openai_api_key'] = api_key
        
        # Show clean status indicator
        if st.session_state.get('api_key_valid'):
            st.caption("🟢 API key verified and ready")
        elif api_key == st.session_state.get('last_verified_key', ''):
            st.caption("❌ API key invalid")
        else:
            st.caption("🟡 Verifying...")
    else:
        st.caption("🔑 Enter API key above")

# Sidebar: Core Methodology
with st.sidebar:
    st.markdown("### **Core Methodology**")
    st.markdown("""
    **Primary Components:**
    
    🔍 **1. Causal Discovery**
    - Identify causal relationships
    - Determine causal direction
    - Build causal graph structure
    
    🎯 **2. Causal Inference** 
    - Estimate treatment effects
    - Calculate confidence intervals
    - Validate causal assumptions
    
    ---
    **Supporting Analysis:**
    
    📊 **Variable Relationships**
    - Correlation analysis
    - Data exploration helper
    
    🤖 **AI Insights**    - Interpretation assistance
    - Policy recommendations
    """)
    
    st.markdown("---")
    st.markdown("**🧪 Scientific Approach:**")
    st.markdown("DirectLiNGAM → DoWhy → AI Analysis")

# Main interface with professional hero section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Causal AI Platform</h1>
    <p class="hero-subtitle">
        Discover causal relationships and quantify treatment effects with advanced AI-powered analysis. 
        Our platform combines cutting-edge algorithms with intelligent domain expertise to uncover hidden patterns, 
        measure intervention impacts, and deliver actionable insights for data-driven decision making.
    </p>
</div>
""", unsafe_allow_html=True)

# Step 1: Data Upload
st.markdown('<div class="step-header"><h2>📁 Step 1: Data Upload</h2></div>', unsafe_allow_html=True)

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
            """)    # Step 2: Domain Constraints (AI-Powered)
    st.markdown('<div class="step-header"><h2>🧠 Step 2: Domain Constraints (AI-Powered)</h2></div>', unsafe_allow_html=True)
    
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
                        st.session_state.get('openai_api_key')                    )
                    
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
        
        # Get user approval/rejection
        approved_constraints = display_simple_constraint_approval(st.session_state['suggested_constraints'])        # Action buttons stacked vertically
        if st.button("Apply Selected", type="primary", key="apply_constraints"):
            st.session_state['constraints_data'] = approved_constraints
            st.session_state['domain_constraints_generated'] = True
            st.session_state['constraints_generated'] = False
            analyzer.domain_constraints = approved_constraints
            st.rerun()
        
        if st.button("Skip All", key="skip_all"):
            st.session_state['constraints_generated'] = False
            st.rerun()
    
    # Manual constraint builder
    with st.expander("Manual Constraints", expanded=False):
        from llm.llm import display_manual_constraint_builder, combine_ai_and_manual_constraints
        
        manual_constraints = display_manual_constraint_builder(list(analyzer.data.columns))
        
        # Store manual constraints in session state for preview
        if manual_constraints['required_edges'] or manual_constraints['forbidden_edges']:
            st.session_state['manual_constraints'] = manual_constraints
            
            # Auto-apply manual constraints if they exist
            if st.session_state.get('domain_constraints_generated') and st.session_state.get('constraints_data'):
                # Combine with existing AI constraints
                ai_constraints = st.session_state['constraints_data']
                combined = combine_ai_and_manual_constraints(ai_constraints, manual_constraints)
                st.session_state['constraints_data'] = combined
                analyzer.domain_constraints = combined
            else:
                # Apply manual constraints only
                st.session_state['constraints_data'] = manual_constraints
                st.session_state['domain_constraints_generated'] = True
                analyzer.domain_constraints = manual_constraints# Show active constraints in a single, clean display
    if st.session_state.get('domain_constraints_generated') and st.session_state.get('constraints_data'):
        constraints_data = st.session_state['constraints_data']
        num_required = len(constraints_data.get('required_edges', []))
        num_forbidden = len(constraints_data.get('forbidden_edges', []))
        
        # Single success message with constraint details
        constraint_details = []
        
        if num_required > 0:
            for edge in constraints_data['required_edges']:
                constraint_details.append(f"✅ **{edge[0]}** causes **{edge[1]}**")
        
        if num_forbidden > 0:
            for edge in constraints_data['forbidden_edges']:
                constraint_details.append(f"🚫 **{edge[0]}** cannot cause **{edge[1]}**")        # Combined display
        if constraint_details:
            st.markdown("**Active Constraints:**")
            # Display each constraint on a separate line
            for detail in constraint_details:
                st.markdown(f"• {detail}")
    # Step 3: Causal Discovery
    st.markdown('<div class="step-header"><h2>🔍 Step 3: Causal Discovery</h2></div>', unsafe_allow_html=True)
    
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
            st.success("✅ Causal discovery completed!")
            
            if active_constraints:
                st.info(f"🧠 Used AI constraints: {len(active_constraints.get('forbidden_edges', []))} forbidden edges, {len(active_constraints.get('required_edges', []))} required edges")
        else:
            st.session_state['causal_discovery_completed'] = False
            st.session_state['last_action'] = 'discovery_failed'
            st.error("❌ Causal discovery failed")
    
    # Show causal discovery results
    if st.session_state['causal_discovery_completed'] and analyzer.adjacency_matrix is not None:
        st.subheader("📈 Discovered Causal Graph")
        
        # Use columns from discovery (these match the adjacency matrix dimensions)
        if hasattr(analyzer.discovery, 'columns') and analyzer.discovery.columns:
            graph_columns = analyzer.discovery.columns
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
        
        show_causal_graph(analyzer.adjacency_matrix, graph_columns)    # Step 4: Variable Relationship Analysis
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
                    st.write("No strong correlations (>0.5) found")# Step 5: Causal Inference Analysis
    st.markdown('<div class="step-header"><h2>🔬 Step 5: Causal Inference Analysis</h2></div>', unsafe_allow_html=True)
    
    # Check if causal discovery has been run
    if analyzer.adjacency_matrix is None:
        st.warning("⚠️ **Causal discovery must be run before causal inference.** Please complete Step 3 first.")
        st.info("💡 **Why this matters:** Causal inference requires understanding the causal structure between variables, which is discovered in Step 3.")
    elif analyzer.data is not None and not analyzer.data.empty:
        # Get columns for causal analysis
        numeric_columns = analyzer.get_numeric_columns()
        categorical_columns = analyzer.get_categorical_columns()
        all_columns = numeric_columns + categorical_columns
        
        if len(all_columns) < 2:
            st.error("❌ Need at least 2 variables for causal analysis. Please ensure your data contains numeric or categorical columns.")
        else:            
            # Show user which variables are available
            if categorical_columns:
                st.info(f"ℹ️ **Variable Selection:** {len(numeric_columns)} numeric + {len(categorical_columns)} categorical variables available. Categorical treatments use specialized policy scenarios.")
            
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
                    st.error("❌ No numeric variables available for outcome. At least one numeric variable is required as the outcome.")
                    st.stop()
                
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
                    help="Select a numeric variable that represents the outcome you want to measure"                )
                  # Store the current outcome selection for next time
                st.session_state['previous_outcome_var'] = outcome_var
            
            if st.button("🔬 Run Causal Inference", type="primary", key="run_causal_inference_btn"):
                if treatment_var != outcome_var:
                    with st.spinner("Running causal inference..."):
                        ate_results = analyzer.calculate_ate(treatment_var, outcome_var)
                        st.session_state['ate_results'] = ate_results
                        st.session_state['selected_treatment'] = treatment_var
                        st.session_state['selected_outcome'] = outcome_var
                        st.session_state['last_action'] = 'inference_completed'
                else:
                    st.error("❌ Please select different variables for treatment and outcome")
    
    # Display results
    if st.session_state.get('ate_results'):
        ate_results = st.session_state['ate_results']
        treatment_var = st.session_state['selected_treatment']
        outcome_var = st.session_state['selected_outcome']
        
        st.success("✅ Causal inference completed!")
          # Main result
        st.subheader("📊 Main Result")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Causal Effect Estimate", 
                f"{ate_results['consensus_estimate']:.4f}"
            )
        
        with col2:
            st.info(f"**Interpretation:** {ate_results['interpretation']}")        # Detailed results        st.subheader("🔍 Detailed Results by Method")
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
          # Step 6: Interactive Policy Explorer - only if we have a meaningful effect
        if abs(ate_results['consensus_estimate']) > 0.01:  # Only show if we have a meaningful effect
            st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', unsafe_allow_html=True)
            st.info("💡 **Explore Policy Scenarios:** Use the estimated causal effect to simulate different intervention strategies.")
            show_interactive_scenario_explorer(ate_results, treatment_var, outcome_var, analyzer)
        else:
            st.markdown('<div class="step-header"><h2>🎮 Step 6: Interactive Policy Explorer</h2></div>', unsafe_allow_html=True)
            st.warning("⚠️ **Policy Explorer unavailable:** The estimated causal effect is too small (≈0) to provide meaningful scenario predictions.")
          # Step 7: AI-Powered Insights
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
