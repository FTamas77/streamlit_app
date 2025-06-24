"""
Core Application Constants for Causal AI Platform
Only truly global constants that affect multiple modules
"""

# ============================================================================
# GLOBAL APPLICATION CONSTANTS
# ============================================================================

# Application metadata
APP_NAME = "Causal AI Platform"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Causal AI Team"

# File handling (affects multiple modules)
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
MAX_FILE_SIZE_MB = 100

# Core data validation (affects multiple modules)
MIN_DATA_ROWS = 10                # Minimum rows required for any analysis
MIN_DATA_COLUMNS = 2              # Minimum columns required for causal analysis

# Global performance limits
COMPUTATION_TIMEOUT_SECONDS = 300 # Maximum time for expensive operations
LARGE_DATASET_THRESHOLD = 10000   # Rows threshold for performance warnings
