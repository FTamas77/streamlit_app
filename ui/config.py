"""
UI Configuration Constants
Constants specific to user interface behavior and display settings
"""

# ============================================================================
# DISPLAY LIMITS AND PAGINATION
# ============================================================================

# Data display limits
MAX_CORRELATION_DISPLAY = 10      # Maximum correlations to show in UI
MAX_CONSTRAINT_DISPLAY = 20       # Maximum constraints to display
MAX_PREVIEW_ROWS = 100            # Maximum rows to show in data preview
MAX_PREVIEW_COLUMNS = 20          # Maximum columns to show in data preview

# Graph visualization
DEFAULT_GRAPH_NODE_SIZE = 20      # Default node size in causal graphs
MAX_GRAPH_DISPLAY_NODES = 25      # Maximum nodes before simplifying display
GRAPH_EDGE_THICKNESS = 2          # Default edge thickness

# ============================================================================
# SESSION AND STATE MANAGEMENT
# ============================================================================

# Session limits
MAX_SESSION_DURATION_HOURS = 24   # Maximum session duration
MAX_ANALYZER_INSTANCES = 10       # Maximum analyzer instances per session
SESSION_CLEANUP_INTERVAL = 3600   # Cleanup interval in seconds

# Cache settings
CACHE_TTL_SECONDS = 1800          # Cache time-to-live (30 minutes)
MAX_CACHE_SIZE_MB = 500           # Maximum cache size per session

# ============================================================================
# RESPONSIVE DESIGN BREAKPOINTS
# ============================================================================

# Screen size breakpoints for responsive design
MOBILE_BREAKPOINT = 768           # Mobile device width threshold
TABLET_BREAKPOINT = 1024          # Tablet device width threshold
DESKTOP_BREAKPOINT = 1200         # Desktop width threshold

# Column layouts for different screen sizes
MOBILE_COLUMNS = 1                # Single column on mobile
TABLET_COLUMNS = 2                # Two columns on tablet  
DESKTOP_COLUMNS = 3               # Three columns on desktop

# ============================================================================
# ANIMATION AND INTERACTION SETTINGS
# ============================================================================

# Animation durations (in milliseconds)
FADE_DURATION = 300               # Standard fade transition
SLIDE_DURATION = 400              # Slide transition duration
LOADING_SPINNER_DELAY = 500       # Delay before showing loading spinner

# Interactive behavior
DEBOUNCE_DELAY = 300              # Input debounce delay in milliseconds
TOOLTIP_DELAY = 800               # Tooltip show delay
AUTO_SAVE_INTERVAL = 30000        # Auto-save interval in milliseconds
