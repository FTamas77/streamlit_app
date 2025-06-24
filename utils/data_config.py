"""
Data Processing Configuration
Constants for data loading, validation, and preprocessing
"""

# ============================================================================
# DATA LOADING SETTINGS
# ============================================================================

# Pandas read settings
PANDAS_READ_SETTINGS = {
    'csv': {
        'encoding': 'utf-8',
        'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN'],
        'keep_default_na': True,
        'skip_blank_lines': True
    },
    'excel': {
        'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN'],
        'keep_default_na': True
    }
}

# Data type inference
AUTO_CONVERT_TYPES = True          # Automatically convert data types
INFER_DATETIME = True              # Try to infer datetime columns
DATE_FORMATS = [                   # Common date formats to try
    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', 
    '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M'
]

# ============================================================================
# DATA VALIDATION RULES
# ============================================================================

# Column name validation
MAX_COLUMN_NAME_LENGTH = 100       # Maximum characters in column names
FORBIDDEN_COLUMN_CHARS = ['/', '\\', '?', '<', '>', ':', '*', '|', '"']
RESERVED_COLUMN_NAMES = ['index', 'level_0', 'level_1']

# Data quality checks
MIN_UNIQUE_VALUES = 2              # Minimum unique values per column (avoid constants)
MAX_CARDINALITY_RATIO = 0.9        # Maximum unique values / total rows ratio
OUTLIER_Z_THRESHOLD = 3.0          # Z-score threshold for outlier detection

# Missing data handling
MAX_MISSING_RATIO = 0.5            # Maximum missing data ratio per column
IMPUTATION_METHODS = {
    'numeric': 'median',           # Default imputation for numeric columns
    'categorical': 'mode'          # Default imputation for categorical columns
}

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Encoding settings
MAX_CATEGORICAL_LEVELS = 50        # Maximum levels before suggesting grouping
ONE_HOT_THRESHOLD = 10             # Maximum levels for one-hot encoding
LABEL_ENCODE_THRESHOLD = 50        # Use label encoding above this threshold

# Scaling and normalization
DEFAULT_SCALER = 'standard'        # Default scaling method
SCALING_METHODS = ['standard', 'minmax', 'robust', 'none']

# Feature engineering
AUTO_CREATE_FEATURES = False       # Automatically create interaction features
MAX_POLYNOMIAL_DEGREE = 2          # Maximum polynomial degree for features
