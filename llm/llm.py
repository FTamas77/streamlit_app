import json
import streamlit as st
from typing import Dict, List

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def generate_domain_constraints(columns: List[str], domain_context: str, api_key: str = None) -> Dict:
    """Use LLM to generate domain-specific constraints with detailed reasoning
    
    Parameters:
    - columns (List[str]): List of dataset column names
    - domain_context (str): Business domain description  
    - api_key (str, optional): OpenAI API key, defaults to None
    
    Returns:
    - Dict: Dictionary containing constraint rules and reasoning for causal discovery
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
        
        Extract ONLY what is explicitly stated in the domain context. For each constraint, provide clear reasoning. 
        
        Generate domain constraints for causal discovery in valid JSON format:
        {{
            "forbidden_edges": [
                {{
                    "edge": ["source", "target"],
                    "reasoning": "Brief explanation of why this relationship should be forbidden"
                }}
            ],
            "required_edges": [
                {{
                    "edge": ["source", "target"], 
                    "reasoning": "Brief explanation of why this relationship must exist"
                }}
            ],
            "explanation": "Overall explanation of the constraints and domain logic"
        }}
        
        **Examples:**
        
        Domain context: "Distance causes emissions; Weather cannot affect vehicle weight"
        Available columns: ["Distance_km", "CO2_Emissions_kg", "Weather_Condition", "Vehicle_Weight_kg"]
        JSON:
        {{
            "forbidden_edges": [
                {{
                    "edge": ["Weather_Condition", "Vehicle_Weight_kg"],
                    "reasoning": "Weather conditions cannot physically change the weight of a vehicle"
                }}
            ],
            "required_edges": [
                {{
                    "edge": ["Distance_km", "CO2_Emissions_kg"],
                    "reasoning": "Longer transportation distances directly increase fuel consumption and emissions"
                }}
            ],
            "explanation": "Transportation logistics constraints: distance drives emissions, but weather doesn't affect vehicle mass"
        }}
        
        IMPORTANT: 
        - Only extract relationships explicitly mentioned
        - Use exact column names from the provided list  
        - Provide specific, domain-relevant reasoning for each constraint
        - If a relationship is mentioned but columns don't match exactly, skip it
        
        Respond only with valid JSON.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        constraints = json.loads(response.choices[0].message.content)
        
        # Ensure the response has the expected structure
        if not isinstance(constraints.get('forbidden_edges'), list):
            constraints['forbidden_edges'] = []
        if not isinstance(constraints.get('required_edges'), list):
            constraints['required_edges'] = []
        
        return constraints
        
    except Exception as e:
        st.error(f"Error generating constraints: {str(e)}")
        return {
            "forbidden_edges": [], 
            "required_edges": [], 
            "explanation": f"Error: {str(e)}"
        }

def display_constraint_approval_interface(suggested_constraints: Dict, columns: List[str]) -> Dict:
    """Display an interactive interface for users to approve/reject/edit LLM-suggested constraints
    
    Parameters:
    - suggested_constraints (Dict): LLM-generated constraints with reasoning
    - columns (List[str]): Available column names for editing
    
    Returns:
    - Dict: User-approved constraints in the standard format
    """
    st.markdown("### 🤖 AI Constraint Suggestions - Your Review Required")
    st.info("💡 **Review each suggestion below.** Accept ✅, reject ❌, or edit ✏️ constraints before applying them to your analysis.")
    
    # Initialize session state for constraint approvals
    if 'constraint_approvals' not in st.session_state:
        st.session_state.constraint_approvals = {}
    
    approved_constraints = {"forbidden_edges": [], "required_edges": []}
    
    # Show overall explanation
    if suggested_constraints.get('explanation'):
        with st.expander("🧠 AI's Overall Reasoning", expanded=False):
            st.markdown(suggested_constraints['explanation'])
    
    # Process required edges
    if suggested_constraints.get('required_edges'):
        st.markdown("#### ✅ **Required Relationships** (AI suggests these MUST exist)")
        
        for i, constraint in enumerate(suggested_constraints['required_edges']):
            # Handle both old format [source, target] and new format with reasoning
            if isinstance(constraint, dict):
                edge = constraint['edge']
                reasoning = constraint.get('reasoning', 'No reasoning provided')
            else:
                edge = constraint
                reasoning = 'Legacy constraint without reasoning'
            
            source, target = edge[0], edge[1]
            constraint_key = f"required_{i}_{source}_{target}"
            
            # Create approval interface
            col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
            
            with col1:
                st.markdown(f"**{source} → {target}**")
            
            with col2:
                st.markdown(f"💭 *{reasoning}*")
            
            with col3:
                approval = st.selectbox(
                    "Action",
                    ["Review", "✅ Accept", "❌ Reject", "✏️ Edit"],
                    key=f"approval_{constraint_key}",
                    label_visibility="collapsed"
                )
            
            with col4:
                if approval == "✏️ Edit":
                    if st.button("Edit", key=f"edit_btn_{constraint_key}"):
                        st.session_state[f"editing_{constraint_key}"] = True
            
            # Handle editing mode
            if st.session_state.get(f"editing_{constraint_key}", False):
                with st.container():
                    st.markdown("**✏️ Edit Constraint:**")
                    edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 1])
                    
                    with edit_col1:
                        new_source = st.selectbox(
                            "New Source", 
                            columns, 
                            index=columns.index(source) if source in columns else 0,
                            key=f"edit_source_{constraint_key}"
                        )
                    
                    with edit_col2:
                        new_target = st.selectbox(
                            "New Target", 
                            columns, 
                            index=columns.index(target) if target in columns else 0,
                            key=f"edit_target_{constraint_key}"
                        )
                    
                    with edit_col3:
                        if st.button("💾 Save", key=f"save_{constraint_key}"):
                            # Update the constraint
                            suggested_constraints['required_edges'][i] = {
                                'edge': [new_source, new_target],
                                'reasoning': f"User-edited: {reasoning}"
                            }
                            st.session_state[f"editing_{constraint_key}"] = False
                            st.rerun()
                        
                        if st.button("❌ Cancel", key=f"cancel_{constraint_key}"):
                            st.session_state[f"editing_{constraint_key}"] = False
                            st.rerun()
            
            # Add to approved constraints if accepted
            if approval == "✅ Accept":
                approved_constraints["required_edges"].append(edge)
            
            st.markdown("---")
    
    # Process forbidden edges
    if suggested_constraints.get('forbidden_edges'):
        st.markdown("#### 🚫 **Forbidden Relationships** (AI suggests these CANNOT exist)")
        
        for i, constraint in enumerate(suggested_constraints['forbidden_edges']):
            # Handle both old format and new format with reasoning
            if isinstance(constraint, dict):
                edge = constraint['edge']
                reasoning = constraint.get('reasoning', 'No reasoning provided')
            else:
                edge = constraint
                reasoning = 'Legacy constraint without reasoning'
            
            source, target = edge[0], edge[1]
            constraint_key = f"forbidden_{i}_{source}_{target}"
            
            # Create approval interface
            col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
            
            with col1:
                st.markdown(f"**{source} ↛ {target}**")
            
            with col2:
                st.markdown(f"💭 *{reasoning}*")
            
            with col3:
                approval = st.selectbox(
                    "Action",
                    ["Review", "✅ Accept", "❌ Reject", "✏️ Edit"],
                    key=f"approval_{constraint_key}",
                    label_visibility="collapsed"
                )
            
            with col4:
                if approval == "✏️ Edit":
                    if st.button("Edit", key=f"edit_btn_{constraint_key}"):
                        st.session_state[f"editing_{constraint_key}"] = True
            
            # Handle editing mode (similar to required edges)
            if st.session_state.get(f"editing_{constraint_key}", False):
                with st.container():
                    st.markdown("**✏️ Edit Constraint:**")
                    edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 1])
                    
                    with edit_col1:
                        new_source = st.selectbox(
                            "New Source", 
                            columns, 
                            index=columns.index(source) if source in columns else 0,
                            key=f"edit_source_{constraint_key}"
                        )
                    
                    with edit_col2:
                        new_target = st.selectbox(
                            "New Target", 
                            columns, 
                            index=columns.index(target) if target in columns else 0,
                            key=f"edit_target_{constraint_key}"
                        )
                    
                    with edit_col3:
                        if st.button("💾 Save", key=f"save_{constraint_key}"):
                            # Update the constraint
                            suggested_constraints['forbidden_edges'][i] = {
                                'edge': [new_source, new_target],
                                'reasoning': f"User-edited: {reasoning}"
                            }
                            st.session_state[f"editing_{constraint_key}"] = False
                            st.rerun()
                        
                        if st.button("❌ Cancel", key=f"cancel_{constraint_key}"):
                            st.session_state[f"editing_{constraint_key}"] = False
                            st.rerun()
            
            # Add to approved constraints if accepted
            if approval == "✅ Accept":
                approved_constraints["forbidden_edges"].append(edge)
            
            st.markdown("---")
    
    # Bulk actions
    st.markdown("#### 🎛️ Bulk Actions")
    bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
    
    with bulk_col1:
        if st.button("✅ Accept All", key="accept_all_constraints"):
            # Auto-accept all constraints
            for constraint in suggested_constraints.get('required_edges', []):
                edge = constraint['edge'] if isinstance(constraint, dict) else constraint
                approved_constraints["required_edges"].append(edge)
            
            for constraint in suggested_constraints.get('forbidden_edges', []):
                edge = constraint['edge'] if isinstance(constraint, dict) else constraint
                approved_constraints["forbidden_edges"].append(edge)
            
            st.success("✅ All constraints accepted!")
    
    with bulk_col2:
        if st.button("❌ Reject All", key="reject_all_constraints"):
            approved_constraints = {"forbidden_edges": [], "required_edges": []}
            st.info("❌ All constraints rejected")
    
    with bulk_col3:
        # Show summary of current selections
        total_required = len(suggested_constraints.get('required_edges', []))
        total_forbidden = len(suggested_constraints.get('forbidden_edges', []))
        accepted_required = len(approved_constraints["required_edges"])
        accepted_forbidden = len(approved_constraints["forbidden_edges"])
        
        st.metric(
            "Progress", 
            f"{accepted_required + accepted_forbidden}/{total_required + total_forbidden}",
            f"{accepted_required} req + {accepted_forbidden} forb"
        )
    
    return approved_constraints

def display_simple_constraint_approval(suggested_constraints: Dict) -> Dict:
    """Clean, professional constraint approval interface for academic use"""
    import streamlit as st
    
    st.markdown("### 🤖 AI-Generated Constraints")
    
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

def explain_results_with_llm(ate_results: Dict, treatment: str, outcome: str, api_key: str = None) -> str:
    """Use LLM to explain causal analysis results"""
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
            return "No API key provided"
        
        openai.api_key = effective_api_key
        
        prompt = f"""
        Explain these causal analysis results in business terms:
        
        Treatment: {treatment}
        Outcome: {outcome}
        Average Treatment Effect: {ate_results.get('consensus_estimate', 'N/A')}
        
        Methods and results:
        {json.dumps(ate_results, indent=2)}
        
        Provide:
        1. Plain English interpretation
        2. Business implications
        3. Confidence assessment
        4. Actionable recommendations
        
        Keep it practical and non-technical.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def display_manual_constraint_builder(columns: List[str]) -> Dict:
    """Professional manual constraint builder interface"""
    import streamlit as st
    
    # Initialize session state for manual constraints if not exists
    if 'manual_constraints' not in st.session_state:
        st.session_state['manual_constraints'] = {"forbidden_edges": [], "required_edges": []}
    
    st.markdown("### Manual Constraints")
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
            if relationship == "→":
                constraint = [cause_var, effect_var]
                if constraint not in st.session_state['manual_constraints']['required_edges']:
                    st.session_state['manual_constraints']['required_edges'].append(constraint)
            else:  # ↛
                constraint = [cause_var, effect_var]
                if constraint not in st.session_state['manual_constraints']['forbidden_edges']:
                    st.session_state['manual_constraints']['forbidden_edges'].append(constraint)
    
    # Display current manual constraints - minimal design
    manual_constraints = st.session_state['manual_constraints']
      # Required edges
    if manual_constraints['required_edges']:
        st.markdown("**Required:**")
        for i, edge in enumerate(manual_constraints['required_edges']):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.write(f"{edge[0]} → {edge[1]}")
            with col2:
                if st.button("×", key=f"delete_req_{i}"):
                    st.session_state['manual_constraints']['required_edges'].pop(i)
                    st.rerun()
    
    # Forbidden edges
    if manual_constraints['forbidden_edges']:
        st.markdown("**Forbidden:**")
        for i, edge in enumerate(manual_constraints['forbidden_edges']):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.write(f"{edge[0]} ↛ {edge[1]}")
            with col2:
                if st.button("×", key=f"delete_forb_{i}"):
                    st.session_state['manual_constraints']['forbidden_edges'].pop(i)
                    st.rerun()
    
    return manual_constraints

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
