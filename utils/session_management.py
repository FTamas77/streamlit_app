"""
Session State Management for Causal AI Platform
Handles initialization and management of Streamlit session state variables
"""

import streamlit as st
from causal_ai.analyzer import CausalAnalyzer

def get_analyzer() -> CausalAnalyzer:
    """Get or create analyzer instance for this user session"""
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = CausalAnalyzer()
    return st.session_state['analyzer']

def init_session_state() -> None:
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

def reset_analysis_state():
    """Reset analysis-related session state (useful when new data is loaded)"""
    reset_keys = [
        'causal_discovery_completed',
        'ate_results', 
        'selected_treatment',
        'selected_outcome',
        'step4_completed',
        'step4_relationships',
        'traditional_results',
        'comparison_results',
        'domain_constraints_generated',
        'constraints_data'
    ]
    
    for key in reset_keys:
        if key in st.session_state:
            if key == 'constraints_data':
                st.session_state[key] = None
            else:
                st.session_state[key] = False if isinstance(st.session_state[key], bool) else None
