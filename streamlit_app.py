# CRITICAL UNDERSTANDING: This entire script runs from TOP to BOTTOM on:
# 1. ✅ User opens the page (first visit)
# 2. ✅ User clicks ANY button 
# 3. ✅ User uploads a file
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

from causal.analyzer import CausalAnalyzer
from llm.llm import generate_domain_constraints, explain_results_with_llm
from ui.components import show_data_preview, show_data_quality_summary, show_correlation_heatmap, show_causal_graph, show_results_table

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

# Simple session state initialization - much cleaner than state machine
if 'causal_discovery_completed' not in st.session_state:
    st.session_state['causal_discovery_completed'] = False

if 'selected_treatment' not in st.session_state:
    st.session_state['selected_treatment'] = None

if 'selected_outcome' not in st.session_state:
    st.session_state['selected_outcome'] = None

if 'ate_results' not in st.session_state:
    st.session_state['ate_results'] = None

if 'domain_constraints_generated' not in st.session_state:
    st.session_state['domain_constraints_generated'] = False

if 'constraints_data' not in st.session_state:
    st.session_state['constraints_data'] = None

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Required for LLM-guided domain constraints and insights",
        value=st.session_state.get('openai_api_key', '')
    )
    
    if api_key:
        st.session_state['openai_api_key'] = api_key
        st.success("✅ API Key saved for this session")
    
    st.markdown("### 📋 Analysis Steps")
    st.markdown("""
    1. Upload Excel/CSV file
    2. Configure domain constraints
    3. Run causal discovery
    4. Calculate treatment effects
    5. Get AI explanations
    """)

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
uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file",
    type=['xlsx', 'csv'],
    help="Ensure your data has clean column names and numeric values."
)

if uploaded_file:
    if analyzer.load_data(uploaded_file):
        # 🚦 STATE TRANSITION: Data uploaded
        st.session_state['data_loaded'] = True
        st.session_state['causal_discovery_completed'] = False
        
        st.success("✅ Data loaded successfully!")
        
        # Display data preview
        show_data_preview(analyzer.data)
        show_data_quality_summary(analyzer.data)
        
        # Step 2: Domain Constraints
        st.markdown('<div class="step-header"><h2>🧠 Step 2: Domain Constraints (AI-Powered)</h2></div>', unsafe_allow_html=True)
        
        domain_context = st.text_area(
            "Describe your domain/business context:",
            placeholder="e.g., 'This is customer behavior data where age comes before purchase decisions...'",
            height=100
        )
        
        if st.button("🤖 Generate Domain Constraints", type="secondary"):
            if domain_context:
                with st.spinner("AI is analyzing your domain..."):
                    constraints = generate_domain_constraints(
                        list(analyzer.data.columns), 
                        domain_context,
                        st.session_state.get('openai_api_key')
                    )
                
                if constraints and constraints.get('explanation') != "No API key provided":
                    st.success("✅ Domain constraints generated!")
                    st.session_state['constraints_data'] = constraints
                    st.session_state['domain_constraints_generated'] = True
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(constraints)
                    with col2:
                        st.info(f"**Explanation:** {constraints.get('explanation', 'No explanation provided')}")
                    
                    analyzer.domain_constraints = constraints
                else:
                    st.error("❌ Failed to generate constraints. Check your API key.")
            else:
                st.warning("Please provide domain context first.")
        
        # Show previously generated constraints
        if st.session_state.get('domain_constraints_generated') and st.session_state.get('constraints_data'):
            with st.expander("📋 Current Domain Constraints", expanded=False):
                st.json(st.session_state['constraints_data'])
        
        # Step 3: Causal Discovery
        st.markdown('<div class="step-header"><h2>🔍 Step 3: Causal Discovery</h2></div>', unsafe_allow_html=True)
        
        # Add validation for constraints
        constraints_ready = (
            st.session_state.get('domain_constraints_generated') or 
            st.checkbox("Skip AI constraints (use default)", help="Run discovery without AI-generated constraints")
        )
        
        if st.button("🚀 Run Causal Discovery", type="primary", disabled=not constraints_ready):
            with st.spinner("Discovering causal relationships..."):
                active_constraints = st.session_state.get('constraints_data', {})
                success = analyzer.run_causal_discovery(active_constraints)
                
            if success:
                st.session_state['causal_discovery_completed'] = True
                st.success("✅ Causal discovery completed!")
                
                if active_constraints:
                    st.info(f"🧠 Used AI constraints: {len(active_constraints.get('forbidden_edges', []))} forbidden edges, {len(active_constraints.get('required_edges', []))} required edges")
            else:
                st.session_state['causal_discovery_completed'] = False
                st.error("❌ Causal discovery failed")
        
        # Show causal discovery results
        if st.session_state['causal_discovery_completed'] and analyzer.adjacency_matrix is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Discovered Causal Graph")
                show_causal_graph(analyzer.adjacency_matrix, list(analyzer.data.columns))
            
            with col2:
                st.subheader("📊 Adjacency Matrix")
                adj_df = pd.DataFrame(
                    analyzer.adjacency_matrix, 
                    index=analyzer.data.columns, 
                    columns=analyzer.data.columns
                )
                st.dataframe(adj_df.round(3))
        
        # Step 4: Variable Relationship Analysis
        st.markdown('<div class="step-header"><h2>📊 Step 4: Variable Relationship Analysis</h2></div>', unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Variable Relationships", type="secondary"):
            with st.spinner("Analyzing variable relationships..."):
                relationships = analyzer.analyze_variable_relationships()
                
                if relationships:
                    st.success("✅ Relationship analysis completed!")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("🔥 Correlation Heatmap")
                        show_correlation_heatmap(relationships['correlation_matrix'])
                    
                    with col2:
                        st.subheader("💪 Strongest Correlations")
                        strong_corr = relationships['strong_correlations']
                        if strong_corr:
                            for corr in strong_corr[:5]:
                                st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
                        else:
                            st.write("No strong correlations (>0.5) found")
        
        # Step 5: Causal Inference Analysis
        st.markdown('<div class="step-header"><h2>🔬 Step 5: Causal Inference Analysis</h2></div>', unsafe_allow_html=True)
        
        if analyzer.data is not None and not analyzer.data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                treatment_var = st.selectbox(
                    "Treatment Variable (Cause)",
                    options=list(analyzer.data.columns),
                    key="treatment_select"
                )
            
            with col2:
                outcome_var = st.selectbox(
                    "Outcome Variable (Effect)", 
                    options=[col for col in analyzer.data.columns if col != treatment_var],
                    key="outcome_select"
                )
            
            with col3:
                confounders = st.multiselect(
                    "Confounding Variables",
                    options=[col for col in analyzer.data.columns if col not in [treatment_var, outcome_var]],
                    key="confounders_select"
                )
            
            if st.button("🔬 Run Causal Inference", type="primary"):
                if treatment_var != outcome_var:
                    with st.spinner("Running causal inference..."):
                        ate_results = analyzer.calculate_ate(treatment_var, outcome_var, confounders)
                        st.session_state['ate_results'] = ate_results
                        st.session_state['selected_treatment'] = treatment_var
                        st.session_state['selected_outcome'] = outcome_var
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
                st.info(f"**Interpretation:** {ate_results['interpretation']}")
            
            # Detailed results
            st.subheader("🔍 Detailed Results by Method")
            show_results_table(ate_results)
            
            # AI Explanation - IMPROVED INTEGRATION
            st.markdown('<div class="step-header"><h2>🧠 AI-Powered Insights</h2></div>', unsafe_allow_html=True)
            
            # Show button only if API key is available
            if st.session_state.get('openai_api_key'):
                if st.button("🤖 Get Detailed AI Analysis", type="secondary"):
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
    """)
