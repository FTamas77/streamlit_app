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

# Comprehensive CSS with Advanced Micro-Interactions
st.markdown("""
<style>
    /* Page fade-in animation */
    .main {
        animation: pageLoad 0.8s ease-out;
    }
    
    @keyframes pageLoad {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Enhanced Hero Section with Micro-Interactions */
    .hero-section {
        text-align: center; 
        padding: 1.5rem 2rem; 
        background: linear-gradient(135deg, #004c6d 0%, #427aa1 50%, #6ca0dc 100%); 
        border-radius: 20px; 
        margin-bottom: 3rem; 
        color: white;
        box-shadow: 0 15px 40px rgba(0, 76, 109, 0.4), 0 5px 15px rgba(0, 76, 109, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .hero-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 50px rgba(0, 76, 109, 0.5), 0 8px 20px rgba(0, 76, 109, 0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(108, 160, 220, 0.2) 0%, transparent 50%);
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    
    .hero-section:hover::before {
        opacity: 0.8;
    }
        .hero-section::after {
        content: '';
        position: absolute;
        top: -2px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent 20%, #6ca0dc 50%, transparent 80%);
        opacity: 0.6;
        border-radius: 20px 20px 0 0;
        z-index: 3;
    }      .hero-title {
        font-size: 2.8rem; 
        font-weight: 800; 
        margin: 0 auto 1rem auto; 
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.03em;
        color: white;
        line-height: 1.1;
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff 0%, #e3f2fd 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleShine 3s ease-in-out infinite;
        transition: transform 0.3s ease;
        text-align: center;
        width: 100%;
        display: block;
    }
    
    .hero-section:hover .hero-title {
        transform: scale(1.02);
    }
      @keyframes titleShine {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }      .hero-subtitle {
        font-size: 1.3rem; 
        font-weight: 400; 
        margin: 0 auto 1.2rem auto; 
        opacity: 0.95; 
        line-height: 1.6;
        max-width: 85%;
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 2;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        transition: opacity 0.3s ease;
        width: 100%;
        display: block;
    }
    
    .hero-section:hover .hero-subtitle {
        opacity: 1;
    }
    
    .hero-subtitle strong {
        color: #e3f2fd;
        font-weight: 600;
    }    /* Enhanced Step Headers with Completion States */    .step-header {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        padding: 1.3rem 1.8rem;
        border-left: 5px solid #3478bd;
        border-radius: 0 12px 12px 0;
        margin: 2.5rem 0 1.8rem 0;
        box-shadow: 0 6px 20px rgba(52, 120, 189, 0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        cursor: pointer;
        border-top: 1px solid rgba(52, 120, 189, 0.2);
        border-bottom: 1px solid rgba(52, 120, 189, 0.2);
    }
      .step-header:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(52, 120, 189, 0.35);
        background: linear-gradient(135deg, #e8effc 0%, #d5e4f7 100%);
        border-left-color: #2563eb;
    }
    
    .step-header.step-completed {
        background: linear-gradient(135deg, #d1f2eb 0%, #a3e3d0 100%);
        border-left-color: #16a085;
        box-shadow: 0 6px 20px rgba(22, 160, 133, 0.3);
    }
    
    .step-header.step-completed:hover {
        background: linear-gradient(135deg, #d8f5ee 0%, #b0e7d5 100%);
        border-left-color: #0e8f7a;
        box-shadow: 0 8px 30px rgba(22, 160, 133, 0.4);
    }
      .step-header.step-completed::after {
        content: '✓';
        position: absolute;
        top: 50%;
        right: 1.8rem;
        transform: translateY(-50%);
        font-size: 1.8rem;
        color: #16a085;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(22, 160, 133, 0.3);
    }
    
    .step-header h2 {
        margin: 0;
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.6rem;
        transition: color 0.3s ease;
        position: relative;
        z-index: 2;
    }
    
    .step-header:hover h2 {
        color: #1a252f;
    }
    
    .step-header.step-completed h2 {
        color: #0d5f4e;
    }

    /* Enhanced Cards and Interactive Elements */
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .professional-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(108, 160, 220, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .professional-card:hover::before {
        left: 100%;
    }
    
    .professional-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #6ca0dc;
    }    /* Enhanced Buttons with Minimal Micro-Interactions */
    .stButton > button {
        background: linear-gradient(135deg, #4f86f7 0%, #6bcff6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.2);
        text-transform: none;
        letter-spacing: 0.3px;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 134, 247, 0.3);
        background: linear-gradient(135deg, #5f96ff 0%, #7bdcff 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        transition: all 0.15s ease;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.3);
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
        color: #64748b;
        cursor: not-allowed;
        transform: none;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.2);
    }
    
    .stButton > button:disabled:hover {
        transform: none;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.2);
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
        border-color: transparent;
    }    /* Enhanced Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
    }
    
    /* Enhanced File Uploader */
    .stFileUploader > div > div {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stFileUploader > div > div:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    /* Enhanced Selectbox and Text Area */
    .stSelectbox > div > div {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:hover {
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.2);
        transform: translateY(-1px);
    }
    
    .stTextArea > div > div > textarea {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stTextArea > div > div > textarea:focus {
        box-shadow: 0 6px 20px rgba(79, 134, 247, 0.3);
        transform: scale(1.01);
    }

    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        background-color: rgba(79, 134, 247, 0.1);
    }

    /* Enhanced Metrics */
    .stMetric {
        transition: all 0.3s ease;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stMetric:hover {
        background-color: rgba(79, 134, 247, 0.05);
        transform: scale(1.02);
    }

    /* Enhanced Expanders */
    .streamlit-expanderHeader {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(79, 134, 247, 0.1);
        transform: translateX(5px);
    }    /* Column hover effects */
    .stColumn {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stColumn:hover {
        background-color: rgba(79, 134, 247, 0.02);
    }

    /* Info/Warning/Error box enhancements */
    .stAlert {
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        transform: scale(1.01);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Dataframe styling */
    .stDataFrame {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stDataFrame:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize analyzer per user session
def get_analyzer():
    """Get or create analyzer instance for this user session"""
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = CausalAnalyzer()
    return st.session_state['analyzer']

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

# Main interface with enhanced hero section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Causal AI Platform</h1>
    <p class="hero-subtitle">
        Discover <strong>causal relationships</strong> and quantify <strong>treatment effects</strong> with advanced AI-powered analysis.<br>
        Transform your data into actionable insights with cutting-edge algorithms and intelligent domain expertise.
    </p>
</div>
""", unsafe_allow_html=True)



# Step 1: Data Upload
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
                        st.session_state.get('openai_api_key')                    )
                    
                    if suggested_constraints and (suggested_constraints.get('forbidden_edges') or suggested_constraints.get('required_edges')):
                        st.session_state['suggested_constraints'] = suggested_constraints
                        st.session_state['constraints_generated'] = True
                    else:
                        st.warning("No specific constraints extracted from context")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Domain context required")    # Show simple approval interface if constraints were generated
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
                st.rerun()      # Manual constraint builder
    with st.expander("➕ Add Manual Constraints", expanded=False):
        from llm.llm import display_manual_constraint_builder
        
        st.markdown("**Add your own domain knowledge:**")
        display_manual_constraint_builder(list(analyzer.data.columns))# Show unified constraint review section
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
            st.info("No constraints in pool. Add some above or skip to use defaults.")    # Step 3: Causal Discovery
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
                    st.write("No strong correlations (>0.5) found")    # Step 5: Causal Inference Analysis
    st.markdown('<div class="step-header"><h2>🔬 Step 5: Causal Inference Analysis</h2></div>', unsafe_allow_html=True)
    
    # Check if causal discovery has been run
    if analyzer.adjacency_matrix is None:
        st.warning("⚠️ **Causal discovery must be run before causal inference.** Please complete Step 3 first.")
        st.info("💡 **Why this matters:** Causal inference requires understanding the causal structure between variables, which is discovered in Step 3.")
    elif analyzer.data is not None and not analyzer.data.empty:        # Get columns for causal analysis
        numeric_columns = analyzer.get_numeric_columns()
        categorical_columns = analyzer.get_categorical_columns()
        all_columns = numeric_columns + categorical_columns
        
        # Debug: Show what columns are detected
        st.info(f"🔢 **Numeric columns** ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None found'}")
        if categorical_columns:
            with st.expander("📊 Categorical columns", expanded=False):
                st.write(f"**{len(categorical_columns)} categorical variables:** {', '.join(categorical_columns)}")
        
        if len(all_columns) < 2:
            st.error("❌ Need at least 2 variables for causal analysis. Please ensure your data contains numeric or categorical columns.")
        else:            
            # Show user which variables are available
            if categorical_columns:
                st.info(f"ℹ️ **Variable Selection:** {len(numeric_columns)} numeric + {len(categorical_columns)} categorical variables available. Categorical treatments use specialized policy scenarios.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                treatment_var = st.selectbox(                    "Treatment Variable (Cause)",
                    options=all_columns,
                    key="treatment_select",
                    help="Select a variable that represents the intervention or treatment (both numeric and categorical treatments are supported)"
                )
            
            with col2:
                # TEMPORAL SOLUTION: Currently restricting outcomes to numeric variables only
                # FUTURE DEVELOPMENT NEEDED: DoWhy supports categorical outcomes for causal inference
                # Research areas for enhancement:
                # 1. Binary outcomes (e.g., success/failure, hired/not hired) using logistic regression
                # 2. Multi-class categorical outcomes using multinomial models
                # 3. Ordinal outcomes with appropriate effect measures
                # 
                # 2. Different effect measures for categorical outcomes (risk ratios, odds ratios, etc.)
                # See DoWhy documentation: https://py-why.github.io/dowhy/
                
                # Outcome should be numeric for meaningful measurement (current limitation)
                available_outcomes = [col for col in numeric_columns if col != treatment_var]
                if not available_outcomes:
                    st.error(f"❌ **No numeric outcome variables available.** Selected treatment: '{treatment_var}' ({'numeric' if treatment_var in numeric_columns else 'categorical'})")
                    st.info("💡 **Solution:** Select a different treatment variable or ensure your data has numeric outcome variables.")
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
                    help="Select a numeric variable that represents the outcome you want to measure"
                )
                
                # Store the current outcome selection for next time
                st.session_state['previous_outcome_var'] = outcome_var
            
            if st.button("🔬 Run Causal Inference", type="primary", key="run_causal_inference_btn"):
                if treatment_var != outcome_var:
                    if outcome_var not in numeric_columns:
                        st.error(f"❌ **Outcome variable '{outcome_var}' must be numeric.**")
                        st.stop()
                    
                    with st.spinner("Running causal inference..."):
                        try:
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
      # Display results
    if st.session_state.get('ate_results'):
        ate_results = st.session_state['ate_results']
        treatment_var = st.session_state['selected_treatment']
        outcome_var = st.session_state['selected_outcome']
          # Main result
        st.subheader("📊 Main Result")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Add effect size context to the metric
            effect_size = abs(ate_results.get('consensus_estimate', 0))
            if effect_size > 1:
                effect_label = "Large Effect"
            elif effect_size > 0.5:
                effect_label = "Moderate Effect"
            elif effect_size > 0.1:
                effect_label = "Small Effect"
            else:
                effect_label = "Very Small Effect"
                
            st.metric(
                f"Causal Effect Estimate ({effect_label})", 
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
