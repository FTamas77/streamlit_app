"""
Configuration module for Causal AI Platform
Centralized imports for global configuration only

Module-specific configurations are kept in their respective modules:
- causal_ai/config.py - Analysis and statistical constants
- ui/config.py - UI behavior and display settings  
- utils/data_config.py - Data processing settings
"""

# Import only global constants that affect multiple modules
from .constants import (
    APP_NAME, APP_VERSION, APP_AUTHOR,
    ALLOWED_FILE_TYPES, MAX_FILE_SIZE_MB,
    MIN_DATA_ROWS, MIN_DATA_COLUMNS,
    COMPUTATION_TIMEOUT_SECONDS, LARGE_DATASET_THRESHOLD
)

# Import UI content and templates (text that may change)
from .ui_content import (
    HERO_SECTION, STEP_TITLES, STEP_DESCRIPTIONS,
    MESSAGES, HELP_TEXT, ERROR_MESSAGES
)

# Import sample data configurations (dataset metadata)
from .sample_data import (
    SAMPLE_DATASETS, DATASET_CATEGORIES, DATA_GENERATION_CONFIG
)

# Make key global items easily accessible
__all__ = [
    # Global app constants
    'APP_NAME', 'APP_VERSION', 'ALLOWED_FILE_TYPES', 'MAX_FILE_SIZE_MB',
    'MIN_DATA_ROWS', 'MIN_DATA_COLUMNS',
    
    # UI content
    'HERO_SECTION', 'STEP_TITLES', 'MESSAGES', 'HELP_TEXT', 'ERROR_MESSAGES',
    
    # Sample data
    'SAMPLE_DATASETS', 'DATASET_CATEGORIES'
]

# Note: Module-specific constants should be imported directly from their modules:
# from causal_ai.config import EFFECT_SIZE_THRESHOLDS, CORRELATION_THRESHOLD
# from ui.config import MAX_CORRELATION_DISPLAY, DEFAULT_GRAPH_NODE_SIZE  
# from utils.data_config import PANDAS_READ_SETTINGS, MAX_MISSING_RATIO
