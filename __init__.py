"""
Causal AI Platform Package

This __init__.py file makes this directory a Python package, enabling:

1. IMPORT FUNCTIONALITY
   - Allows importing modules from subdirectories
   - Example: from causal.analyzer import CausalAnalyzer
   
2. PACKAGE ORGANIZATION  
   - Groups related modules together logically
   - Creates namespace hierarchy (models.*, utils.*, calculations.*)
   
3. EXPLICIT PACKAGE STRUCTURE
   - Makes it clear what directories are packages vs regular folders
   - Python requires __init__.py to treat directories as importable packages
   
4. CENTRALIZED IMPORTS (optional)
   - Can expose key classes/functions at package level
   - Simplifies imports for users
   
5. PACKAGE METADATA
   - Can contain version info, package description
   - Documentation for the overall package

Without __init__.py files:
- Python won't recognize directories as packages
- Cross-module imports would fail
- IDE autocomplete wouldn't work properly
- Package structure would be unclear

Example usage enabled by __init__.py:
```python
# This works because of __init__.py files:
from causal.analyzer import CausalAnalyzer
from utils.data_cleaning import clean_data
from utils.effect_size import classify_effect_size

# Without __init__.py, these imports would fail
```
"""

# Optional: Expose key components at package level
# This allows: from streamlit_causal_ai import CausalAnalyzer
# Instead of: from streamlit_causal_ai.models.causal_analyzer import CausalAnalyzer

__version__ = "1.0.0"
__author__ = "Causal AI Platform Team"
__description__ = "Advanced causal analysis platform with AI integration"

# Expose main components (optional)
try:
    from .models.causal_analyzer import CausalAnalyzer
    from .ai.llm_integration import generate_domain_constraints, explain_results_with_llm
    
    __all__ = [
        'CausalAnalyzer',
        'generate_domain_constraints', 
        'explain_results_with_llm'
    ]
except ImportError:
    # Handle case where dependencies aren't installed
    __all__ = []
