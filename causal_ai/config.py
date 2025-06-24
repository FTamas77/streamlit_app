"""
Analysis Configuration Constants
Constants specific to causal analysis algorithms and statistical methods
"""

# ============================================================================
# STATISTICAL ANALYSIS THRESHOLDS
# ============================================================================

# Effect Size Classification (based on Cohen's conventions)
EFFECT_SIZE_THRESHOLDS = {
    'large': 1.0,
    'moderate': 0.5,
    'small': 0.1
}

# Correlation Analysis
CORRELATION_THRESHOLD = 0.5        # Minimum correlation to be considered "strong"
MIN_POLICY_EFFECT = 0.01          # Minimum effect size to enable policy explorer

# Causal Inference Settings
CONFIDENCE_LEVEL = 0.95           # Default confidence level for causal inference
BOOTSTRAP_SAMPLES = 1000          # Number of bootstrap samples for uncertainty estimation
MAX_ITERATIONS = 10000            # Maximum iterations for causal discovery algorithms

# Data Quality Thresholds
MAX_NULL_PERCENTAGE = 90          # Maximum percentage of null values allowed per column
MAX_CONSTANT_COLUMNS = 0.8        # Maximum fraction of columns that can be constant
MIN_VARIANCE_THRESHOLD = 1e-10    # Minimum variance for numeric columns

# ============================================================================
# CAUSAL DISCOVERY PARAMETERS
# ============================================================================

# Algorithm-specific settings
LINGAM_SETTINGS = {
    'prior_knowledge': None,
    'measure': 'pwling',           # Measure for causal discovery
    'thresh': 0.01                 # Threshold for edge detection
}

# Graph processing
MAX_GRAPH_NODES = 50              # Maximum nodes to display in causal graph
MIN_EDGE_WEIGHT = 0.1             # Minimum edge weight to display

# ============================================================================
# VALIDATION RULES
# ============================================================================

# Variable type validation
NUMERIC_DTYPES = ['int64', 'float64', 'int32', 'float32']
CATEGORICAL_DTYPES = ['object', 'category', 'bool']

# Sample size requirements for different analyses
MIN_SAMPLES_CORRELATION = 30      # Minimum samples for correlation analysis
MIN_SAMPLES_CAUSAL = 100          # Minimum samples for causal inference
MIN_SAMPLES_BOOTSTRAP = 200       # Minimum samples for bootstrap methods
