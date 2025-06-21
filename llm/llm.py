import json
import streamlit as st
from typing import Dict, List

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def generate_domain_constraints(columns: List[str], domain_context: str, api_key: str = None) -> Dict:
    """Use LLM to generate domain-specific constraints
    
    Parameters:
    - columns (List[str]): MUTABLE - List of dataset column names, changes affect original list
    - domain_context (str): IMMUTABLE - Business domain description, behaves like pass-by-value  
    - api_key (str, optional): IMMUTABLE - OpenAI API key, behaves like pass-by-value, defaults to None
    
    Returns:
    - Dict: MUTABLE - Dictionary containing constraint rules for causal discovery
    """
    if not OPENAI_AVAILABLE:
        st.warning("OpenAI not available. Skipping domain constraints generation.")
        return {"forbidden_edges": [], "required_edges": []}
    
    try:
        effective_api_key = api_key
        if not effective_api_key:
            try:
                effective_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                effective_api_key = ""
        
        if not effective_api_key:
            st.error("Please provide an OpenAI API key to use AI features")
            return {"forbidden_edges": [], "required_edges": []}
        
        openai.api_key = effective_api_key
        
        prompt = f"""
        Given a dataset with columns: {', '.join(columns)}
        Domain context: {domain_context}
        
        Generate domain constraints for causal discovery in JSON format:
        {{
            "forbidden_edges": [["source", "target"], ...],
            "required_edges": [["source", "target"], ...]
        }}
        
        Consider:
        1. Temporal relationships (causes must precede effects)
        2. Domain knowledge (what relationships make sense)
        3. Physical/logical impossibilities
        
        Respond only with valid JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        constraints = json.loads(response.choices[0].message.content)
        return constraints
        
    except Exception as e:
        st.error(f"Error generating constraints: {str(e)}")
        return {"forbidden_edges": [], "required_edges": []}

def explain_results_with_llm(ate_results: Dict, treatment: str, outcome: str, api_key: str = None) -> str:
    """Use LLM to explain causal analysis results
    
    Parameters:
    - ate_results (Dict): MUTABLE - Dictionary containing causal analysis results, passed by reference
    - treatment (str): IMMUTABLE - Name of treatment variable, behaves like pass-by-value
    - outcome (str): IMMUTABLE - Name of outcome variable, behaves like pass-by-value
    - api_key (str, optional): IMMUTABLE - OpenAI API key, behaves like pass-by-value, defaults to None
    
    Returns:
    - str: IMMUTABLE - Business-friendly explanation of the causal analysis results
    """
    if not OPENAI_AVAILABLE:
        return "OpenAI package not available. Cannot generate AI explanations."
    
    try:
        effective_api_key = api_key
        if not effective_api_key:
            try:
                effective_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                effective_api_key = ""
        
        if not effective_api_key:
            return f"""
**AI Analysis Unavailable**

**Manual Analysis Summary:**
- Treatment Effect: {ate_results.get('consensus_estimate', 0):.4f}
- This suggests a {'strong' if abs(ate_results.get('consensus_estimate', 0)) > 0.1 else 'moderate'} causal relationship
- Consider validating with controlled experiments
            """
        
        openai.api_key = effective_api_key
        
        prompt = f"""
        Explain the following causal analysis results in business terms:
        
        Treatment: {treatment}
        Outcome: {outcome}
        Average Treatment Effect: {ate_results.get('consensus_estimate', 'N/A')}
        Methods Used: {list(ate_results.get('estimates', {}).keys())}
        Robustness: {ate_results.get('recommendation', 'N/A')}
        
        Provide:
        1. A clear explanation of what this means
        2. Business implications
        3. Reliability of the result
        4. Recommendations for action
        
        Keep it concise and business-friendly.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            return f"""
**API Key Error**
Please check your OpenAI API key.

**Fallback Analysis:**
Your causal analysis shows a treatment effect of {ate_results.get('consensus_estimate', 0):.4f}.
            """
        else:
            return f"""
**AI Analysis Error**
Unable to generate AI explanation due to: {error_msg}

**Manual Summary:**
- Effect Size: {ate_results.get('consensus_estimate', 'N/A'):.4f}
- Interpretation: {ate_results.get('interpretation', 'No interpretation available')}
            """

def display_simple_constraint_approval(suggested_constraints: Dict) -> Dict:
    """Clean, professional constraint approval interface for academic use"""
    import streamlit as st
    
    approved_constraints = {"forbidden_edges": [], "required_edges": []}
    
    # Required edges section - clean and minimal
    required_edges = suggested_constraints.get('required_edges', [])
    if required_edges:
        st.markdown("**Required Relationships:**")
        
        for i, constraint in enumerate(required_edges):
            if isinstance(constraint, dict):
                edge = constraint['edge']
            else:
                edge = constraint
            
            approved = st.checkbox(
                f"{edge[0]} → {edge[1]}",
                value=True,
                key=f"approve_required_{i}"
            )
            
            if approved:
                approved_constraints['required_edges'].append(edge)
    
    # Forbidden edges section - clean and minimal
    forbidden_edges = suggested_constraints.get('forbidden_edges', [])
    if forbidden_edges:
        st.markdown("**Forbidden Relationships:**")
        
        for i, constraint in enumerate(forbidden_edges):
            if isinstance(constraint, dict):
                edge = constraint['edge']
            else:
                edge = constraint
            
            approved = st.checkbox(
                f"{edge[0]} ↛ {edge[1]}",
                value=True,
                key=f"approve_forbidden_{i}"
            )
            
            if approved:
                approved_constraints['forbidden_edges'].append(edge)
    
    # Minimal summary
    total_approved = len(approved_constraints['required_edges']) + len(approved_constraints['forbidden_edges'])
    if total_approved > 0:
        st.caption(f"{total_approved} constraints selected")
    
    return approved_constraints

def get_domain_constraints_help():
    """Return helpful guidance for writing domain constraints"""
    return """
**🎯 Domain constraints help the AI understand your business logic.**

**✅ Good Examples:**
- "Transportation distance increases CO2 emissions"
- "Weather conditions cannot affect vehicle weight"  
- "Fuel type influences emission levels but not distance"

**❌ Avoid:**
- "Everything affects everything" (too vague)
- "Use common sense" (be specific)
- Made-up variable names

**💡 Tips:**
- Be specific about cause → effect relationships
- Mention what CANNOT happen (forbidden relationships)
- Use your actual column names when possible
"""

def display_manual_constraint_builder(columns: List[str]) -> Dict:
    """Professional manual constraint builder interface - immediately adds to pool"""
    import streamlit as st
    
    # Manual constraint input - clean layout
    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
    
    with col1:
        cause_var = st.selectbox("Cause", ["Select..."] + columns, key="manual_cause")
    
    with col2:
        relationship = st.selectbox("Type", ["→", "↛"], key="manual_relationship")
    
    with col3:
        effect_var = st.selectbox("Effect", ["Select..."] + columns, key="manual_effect")
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        add_disabled = cause_var == "Select..." or effect_var == "Select..." or cause_var == effect_var
        
        if st.button("Add", disabled=add_disabled, key="add_manual_constraint"):
            # Immediately add to the main constraint pool
            constraint = [cause_var, effect_var]
            
            if relationship == "→":
                # Add to required edges in the main pool
                if st.session_state.get('constraints_data'):
                    if constraint not in st.session_state['constraints_data'].get('required_edges', []):
                        st.session_state['constraints_data']['required_edges'].append(constraint)
                else:
                    st.session_state['constraints_data'] = {"forbidden_edges": [], "required_edges": [constraint]}
                
                # Mark as having constraints and rerun to show in final list
                st.session_state['domain_constraints_generated'] = True
                st.rerun()
                
            else:  # ↛
                # Add to forbidden edges in the main pool
                if st.session_state.get('constraints_data'):
                    if constraint not in st.session_state['constraints_data'].get('forbidden_edges', []):
                        st.session_state['constraints_data']['forbidden_edges'].append(constraint)
                else:
                    st.session_state['constraints_data'] = {"forbidden_edges": [constraint], "required_edges": []}
                
                # Mark as having constraints and rerun to show in final list
                st.session_state['domain_constraints_generated'] = True
                st.rerun()
    
    # Show simple instructions
    st.info("💡 Use the form above to add manual constraints. They will appear immediately in the constraint review section below.")
    
    # Return empty dict since we're adding directly to the main pool
    return {"forbidden_edges": [], "required_edges": []}

def combine_ai_and_manual_constraints(ai_constraints: Dict, manual_constraints: Dict) -> Dict:
    """Combine AI suggestions with manual constraints, avoiding duplicates"""
    
    combined = {"forbidden_edges": [], "required_edges": []}
    
    # Add AI constraints first
    if ai_constraints:
        combined['forbidden_edges'].extend(ai_constraints.get('forbidden_edges', []))
        combined['required_edges'].extend(ai_constraints.get('required_edges', []))
    
    # Add manual constraints, avoiding duplicates
    for edge in manual_constraints.get('forbidden_edges', []):
        if edge not in combined['forbidden_edges']:
            combined['forbidden_edges'].append(edge)
    
    for edge in manual_constraints.get('required_edges', []):
        if edge not in combined['required_edges']:
            combined['required_edges'].append(edge)
    
    return combined
