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
        return {"forbidden_edges": [], "required_edges": [], "temporal_order": [], "explanation": "OpenAI not available"}
    
    try:
        effective_api_key = api_key
        if not effective_api_key:
            try:
                effective_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                effective_api_key = ""
        
        if not effective_api_key:
            st.error("Please provide an OpenAI API key to use AI features")
            return {"forbidden_edges": [], "required_edges": [], "temporal_order": [], "explanation": "No API key provided"}
        
        openai.api_key = effective_api_key
        
        prompt = f"""
        Given a dataset with columns: {', '.join(columns)}
        Domain context: {domain_context}
        
        Generate domain constraints for causal discovery in JSON format:
        {{
            "forbidden_edges": [["source", "target"], ...],
            "required_edges": [["source", "target"], ...],
            "temporal_order": ["earliest_var", "middle_var", "latest_var", ...],
            "explanation": "Brief explanation of constraints"
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
        return {"forbidden_edges": [], "temporal_order": [], "explanation": "Error generating constraints - check API key"}

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
