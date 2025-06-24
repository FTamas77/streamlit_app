"""
Sidebar Components for Causal AI Platform
Configuration panel and methodology overview
"""

import streamlit as st

def render_api_key_section():
    """Render the API key configuration section"""
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

def render_methodology_section():
    """Render the core methodology overview"""
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
    
    🤖 **AI Insights**
    - Interpretation assistance
    - Policy recommendations
    """)
    
    st.markdown("---")
    st.markdown("**🧪 Scientific Approach:**")
    st.markdown("DirectLiNGAM → DoWhy → AI Analysis")

def render_sidebar():
    """Render the complete sidebar"""
    with st.sidebar:
        render_api_key_section()
        render_methodology_section()
