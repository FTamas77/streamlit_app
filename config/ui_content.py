"""
UI Content and Templates for Causal AI Platform
Text, labels, and content that may change or be localized
"""

# ============================================================================
# HERO SECTION CONTENT
# ============================================================================

HERO_SECTION = {
    'title': 'Causal AI Platform',
    'subtitle': '''Discover <strong>causal relationships</strong> and quantify <strong>treatment effects</strong> 
                   with advanced AI-powered analysis.<br>Transform your data into actionable insights with 
                   cutting-edge algorithms and intelligent domain expertise.'''
}

# ============================================================================
# STEP TITLES AND LABELS
# ============================================================================

STEP_TITLES = {
    1: "📁 Step 1: Data Upload",
    2: "🧠 Step 2: Domain Knowledge & Constraints", 
    3: "🔍 Step 3: Causal Discovery",
    4: "📊 Step 4: Variable Relationship Analysis",
    5: "🔬 Step 5: Causal Inference Analysis",
    6: "🎮 Step 6: Interactive Policy Explorer",
    7: "🧠 Step 7: AI-Powered Insights"
}

STEP_DESCRIPTIONS = {
    1: "Upload your data or select from sample datasets",
    2: "Add domain knowledge to guide causal discovery",
    3: "Discover causal relationships in your data",
    4: "Analyze correlations between variables",
    5: "Estimate causal effects and treatment impacts",
    6: "Explore policy scenarios and interventions",
    7: "Get AI-powered insights and recommendations"
}

# ============================================================================
# USER MESSAGES AND NOTIFICATIONS
# ============================================================================

MESSAGES = {
    'welcome': '''👆 **Get Started:** Upload your Excel or CSV file above to begin causal analysis.
    
**What you can do with this platform:**
- 🔍 Discover causal relationships in your data
- 📊 Calculate treatment effects with confidence intervals  
- 🤖 Get AI-powered insights and recommendations
- 🎯 Simulate policy interventions

**Try our sample dataset:** 🌱 CO2 Supply Chain Analysis - explore how transportation factors affect environmental impact!''',
    
    'discovery_required': '''⚠️ **Causal discovery must be run before causal inference.** Please complete Step 3 first.
💡 **Why this matters:** Causal inference requires understanding the causal structure between variables, which is discovered in Step 3.''',
    
    'policy_unavailable': '⚠️ **Policy Explorer unavailable:** The estimated causal effect is too small (≈0) to provide meaningful scenario predictions.',
    
    'data_upload_success': '✅ Data loaded successfully!',
    'analysis_complete': '🎉 Analysis completed successfully!',
    'api_key_required': '🔑 Add OpenAI API key in sidebar to get AI-powered explanations',
    
    'troubleshooting': '''
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
'''
}

# ============================================================================
# HELP TEXT AND INSTRUCTIONS
# ============================================================================

HELP_TEXT = {
    'file_upload': "Ensure your data has clean column names and numeric values for best results.",
    'domain_constraints': "Describe relationships you know exist or should be forbidden based on domain knowledge.",
    'variable_selection': "Select variables that represent the intervention (treatment) and outcome you want to measure.",
    'graph_reading': '''
**How to read this graph:**
- 🔵 **Nodes** = Variables in your dataset
- ➡️ **Arrows** = Causal relationships (A → B means A causes B)
- 🎨 **Node Colors** = Variable role (Red=Outcome, Teal=Cause, Blue=Mediator)
- 🎛️ **Customize** = Use the controls below to adjust layout and hide relationships'''
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'file_too_large': "File too large ({size:.1f}MB). Maximum size: {max_size}MB",
    'invalid_file_type': "Unsupported file type. Please upload: {allowed_types}",
    'insufficient_data': "Dataset too small ({rows} rows). Minimum {min_rows} rows required for reliable analysis",
    'no_numeric_columns': "Need at least 2 numeric variables for causal analysis. Found {count} numeric columns",
    'same_variables': "Treatment and outcome variables must be different",
    'api_key_invalid': "Invalid API key format. Should start with 'sk-'",
    'analysis_failed': "Analysis encountered an error: {error}"
}
