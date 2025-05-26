# 🔧 API Reference

Complete reference for all classes, methods, and functions in the Causal AI Platform.

## 📋 Table of Contents

- [Core Classes](#core-classes)
- [Utility Functions](#utility-functions)
- [Calculation Modules](#calculation-modules)
- [AI Integration](#ai-integration)
- [UI Components](#ui-components)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## 🏗️ Core Classes

### CausalAnalyzer

Main class for causal analysis pipeline.

```python
from models.causal_analyzer import CausalAnalyzer

analyzer = CausalAnalyzer()
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Cleaned dataset |
| `model` | `DirectLiNGAM` | Causal discovery model |
| `adjacency_matrix` | `np.ndarray` | Causal structure matrix |
| `causal_model` | `CausalModel` | DoWhy causal model |
| `domain_constraints` | `Dict` | AI-generated constraints |

#### Methods

##### `load_data(uploaded_file) -> bool`

Load and clean data from uploaded file.

**Parameters:**
- `uploaded_file` (UploadedFile): Streamlit uploaded file object

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
success = analyzer.load_data(uploaded_file)
if success:
    print(f"Loaded {analyzer.data.shape[0]} rows, {analyzer.data.shape[1]} columns")
```

**Side Effects:**
- Sets `analyzer.data` with cleaned DataFrame
- Displays progress messages via Streamlit
- Handles multiple file formats and encodings

##### `run_causal_discovery(constraints=None) -> bool`

Discover causal relationships in the data.

**Parameters:**
- `constraints` (Dict, optional): Domain constraints from AI or manual input

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
constraints = {
    "forbidden_edges": [["outcome", "treatment"]],
    "required_edges": [["confounder", "outcome"]]
}
success = analyzer.run_causal_discovery(constraints)
```

**Side Effects:**
- Sets `analyzer.adjacency_matrix`
- May set `analyzer.model` if LiNGAM available

##### `calculate_ate(treatment, outcome, confounders=None) -> Dict`

Calculate Average Treatment Effect.

**Parameters:**
- `treatment` (str): Name of treatment variable
- `outcome` (str): Name of outcome variable  
- `confounders` (List[str], optional): List of confounder variable names

**Returns:**
- `Dict`: Comprehensive results dictionary

**Return Structure:**
```python
{
    "estimates": {
        "method_name": {
            "estimate": float,
            "confidence_interval": [float, float],
            "p_value": float,
            "method": str
        }
    },
    "consensus_estimate": float,
    "robustness": Dict,
    "interpretation": str,
    "recommendation": str,
    "additional_metrics": Dict
}
```

**Example:**
```python
results = analyzer.calculate_ate(
    treatment="marketing_spend",
    outcome="sales",
    confounders=["age", "income"]
)
print(f"Causal effect: {results['consensus_estimate']:.4f}")
```

##### `analyze_variable_relationships() -> Dict`

Analyze correlations between all variables.

**Returns:**
- `Dict`: Correlation analysis results

**Return Structure:**
```python
{
    "strong_correlations": [
        {
            "var1": str,
            "var2": str, 
            "correlation": float
        }
    ],
    "correlation_matrix": pd.DataFrame
}
```

##### `analyze_effect_heterogeneity(treatment, outcome, moderator) -> Dict`

Analyze how treatment effects vary by moderator.

**Parameters:**
- `treatment` (str): Treatment variable name
- `outcome` (str): Outcome variable name
- `moderator` (str): Moderating variable name

**Returns:**
- `Dict`: Heterogeneity analysis results

##### `simulate_policy_intervention(treatment, outcome, intervention_size) -> Dict`

Simulate policy intervention effects.

**Parameters:**
- `treatment` (str): Treatment variable name
- `outcome` (str): Outcome variable name
- `intervention_size` (float): Size of intervention

**Returns:**
- `Dict`: Simulation results

## 🛠️ Utility Functions

### Data Cleaning (`utils.data_cleaning`)

#### `clean_data(df) -> pd.DataFrame`

Main data cleaning function.

**Parameters:**
- `df` (pd.DataFrame): Raw input DataFrame

**Returns:**
- `pd.DataFrame`: Cleaned DataFrame or None if cleaning fails

**Cleaning Steps:**
1. Remove empty rows/columns
2. Clean column names
3. Convert to numeric types
4. Handle missing values
5. Remove infinite values

**Example:**
```python
from utils.data_cleaning import clean_data

raw_df = pd.read_csv("messy_data.csv")
clean_df = clean_data(raw_df)
```

#### `_convert_to_numeric(df) -> pd.DataFrame`

Convert DataFrame columns to numeric types.

**Conversion Strategies:**
1. Direct conversion with `pd.to_numeric()`
2. Clean formatting (remove $, %, commas)
3. Extract numbers from strings
4. Convert datetime to timestamps

#### `_handle_missing_values(df) -> pd.DataFrame`

Handle missing values in dataset.

**Strategy:**
- Drop columns with >70% missing values
- Fill remaining missing values with median
- Remove rows with infinite values

### Constraint Utilities (`utils.constraint_utils`)

#### `convert_edge_constraints(edge_list, columns) -> np.ndarray`

Convert edge constraints to adjacency matrix format.

**Parameters:**
- `edge_list` (List[List[str]]): List of forbidden edges
- `columns` (List[str]): Column names

**Returns:**
- `np.ndarray`: Forbidden adjacency matrix

#### `adjacency_to_graph_string(adjacency_matrix, columns) -> str`

Convert adjacency matrix to DoWhy graph string.

**Parameters:**
- `adjacency_matrix` (np.ndarray): Causal adjacency matrix
- `columns` (List[str]): Variable names

**Returns:**
- `str`: Graph string in DoWhy format

**Example:**
```python
graph_string = adjacency_to_graph_string(adj_matrix, columns)
# Returns: '"var1" -> "var2"; "var2" -> "var3"'
```

### Fallback Methods (`utils.fallback_methods`)

#### `correlation_based_discovery(data) -> np.ndarray`

Fallback causal discovery using correlations.

**Parameters:**
- `data` (pd.DataFrame): Input dataset

**Returns:**
- `np.ndarray`: Estimated adjacency matrix

**Algorithm:**
- Calculates correlation matrix
- Uses variance heuristic for direction
- Applies correlation threshold (0.3)

## 📊 Calculation Modules

### Causal Inference (`calculations.causal_inference`)

#### `calculate_ate_dowhy(analyzer, treatment, outcome, confounders) -> Dict`

Calculate ATE using DoWhy library.

**Method:** Backdoor adjustment with linear regression

#### `calculate_ate_fallback(analyzer, treatment, outcome, confounders) -> Dict`

Fallback ATE calculation using sklearn.

**Method:** Linear regression with manual statistical tests

**Statistical Calculations:**
- Treatment effect coefficient
- Standard errors
- T-statistics and p-values
- Confidence intervals

### Metrics (`calculations.metrics`)

#### `classify_effect_size(effect) -> str`

Classify effect magnitude.

**Parameters:**
- `effect` (float): Effect size value

**Returns:**
- `str`: Classification ("Very Small", "Small", "Medium", "Large")

**Thresholds:**
- Very Small: < 0.01
- Small: 0.01 - 0.1  
- Medium: 0.1 - 0.5
- Large: > 0.5

#### `calculate_simple_metrics(data, treatment, outcome, estimate) -> Dict`

Calculate additional statistical metrics.

**Returns:**
```python
{
    "r_squared": float,
    "explained_variance_percent": float,
    "effect_size_interpretation": str
}
```

#### `generate_simple_recommendation(estimates) -> str`

Generate text recommendation based on results.

**Logic:**
- Check statistical significance (p < 0.05)
- Assess practical significance (effect size)
- Provide actionable guidance

#### `interpret_ate(ate_value, treatment, outcome) -> str`

Generate human-readable interpretation of ATE.

**Example:**
```python
interpretation = interpret_ate(0.23, "marketing", "sales")
# Returns: "A one-unit increase in marketing increases sales by 0.23 units on average."
```

### Relationship Analysis (`calculations.relationship_analysis`)

#### `analyze_relationships(data) -> Dict`

Comprehensive relationship analysis.

**Analysis includes:**
- Correlation matrix calculation
- Strong correlation identification (|r| > 0.5)
- Ranking by correlation strength

### Advanced Analysis (`calculations.advanced_analysis`)

#### `analyze_heterogeneity(data, treatment, outcome, moderator) -> Dict`

Effect heterogeneity analysis.

**Method:**
- Median split on moderator variable
- Calculate effects for high/low groups
- Compare effect sizes

#### `simulate_intervention(data, treatment, outcome, intervention_size) -> Dict`

Policy intervention simulation.

**Calculation:**
- Current baseline levels
- Predicted change based on estimated effect
- Percentage impact calculation

## 🤖 AI Integration

### LLM Integration (`ai.llm_integration`)

#### `generate_domain_constraints(columns, domain_context, api_key) -> Dict`

Generate domain constraints using GPT-3.5-turbo.

**Parameters:**
- `columns` (List[str]): Dataset column names
- `domain_context` (str): Business context description
- `api_key` (str, optional): OpenAI API key

**Returns:**
```python
{
    "forbidden_edges": List[List[str]],
    "required_edges": List[List[str]], 
    "temporal_order": List[str],
    "explanation": str
}
```

**API Details:**
- Model: GPT-3.5-turbo
- Temperature: 0.3
- Format: JSON response
- Typical cost: $0.01-0.02 per request

#### `explain_results_with_llm(ate_results, treatment, outcome, api_key) -> str`

Generate business explanation of results.

**Parameters:**
- `ate_results` (Dict): Results from causal inference
- `treatment` (str): Treatment variable name
- `outcome` (str): Outcome variable name
- `api_key` (str, optional): OpenAI API key

**Returns:**
- `str`: Markdown-formatted business explanation

**Explanation includes:**
- Clear interpretation of statistical results
- Business implications
- Reliability assessment  
- Actionable recommendations

## 🎨 UI Components

### Components (`ui.components`)

#### `show_data_preview(data)`

Display data preview with basic metrics.

**Parameters:**
- `data` (pd.DataFrame): Dataset to preview

**UI Elements:**
- Data table (first 5 rows)
- Row/column/missing value metrics

#### `show_data_quality_summary(data)`

Show expandable data quality information.

**Displays:**
- Column data types
- Value ranges
- Basic statistics

#### `show_correlation_heatmap(correlation_matrix)`

Interactive correlation heatmap using Plotly.

**Features:**
- Color coding for correlation strength
- Hover tooltips with exact values
- Responsive design

#### `show_causal_graph(adjacency_matrix, columns)`

Interactive causal graph visualization.

**Features:**
- Directed graph with arrows
- Node labels for variables
- Edge weights displayed
- Interactive hover information

#### `show_results_table(ate_results)`

Formatted results table.

**Columns:**
- Method name
- Point estimate
- Confidence interval
- P-value
- Significance indicator

## ⚙️ Configuration

### Environment Variables

#### Streamlit Secrets

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
DEFAULT_CORRELATION_THRESHOLD = 0.3
MAX_MISSING_PERCENTAGE = 70
```

#### Package Dependencies

**Required:**
```
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.20.0
plotly>=5.0.0
networkx>=2.8.0
scikit-learn>=1.1.0
scipy>=1.9.0
```

**Optional:**
```
lingam>=1.8.0
dowhy>=0.9.0  
openai>=1.0.0
```

### Package Detection

```python
# Automatic package detection
try:
    from lingam import DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False
```

## ⚠️ Error Handling

### Common Exceptions

#### DataLoadingError

**Cause:** File cannot be loaded or parsed
**Handling:** Multiple encoding attempts, format detection
**User Message:** Detailed error with suggestions

#### InsufficientDataError  

**Cause:** < 2 columns or < 10 rows after cleaning
**Handling:** Graceful failure with guidance
**User Message:** Specific requirements and tips

#### CausalDiscoveryError

**Cause:** LiNGAM fails on dataset
**Handling:** Automatic fallback to correlation method
**User Message:** Method switch notification

#### APIError

**Cause:** OpenAI API issues (key, credits, network)
**Handling:** Graceful degradation to manual analysis
**User Message:** API status and fallback options

### Error Recovery Patterns

#### Graceful Degradation

```python
try:
    # Try advanced method
    result = advanced_method(data)
except AdvancedMethodError:
    # Fall back to basic method
    result = basic_method(data)
    st.warning("Used fallback method")
```

#### User Guidance

```python
if error_occurred:
    st.error("❌ Specific error description")
    st.info("💡 Suggestions to fix the issue")
    st.code("# Example of correct format")
```

## 🔄 Session State Management

### Key Session Variables

```python
# Streamlit session state keys
'causal_discovery_completed': bool
'selected_treatment': str  
'selected_outcome': str
'ate_results': Dict
'openai_api_key': str
'run_analysis_executed': bool
```

### State Flow

1. **Data Upload** → Updates analyzer.data
2. **Causal Discovery** → Sets discovery_completed = True
3. **Variable Selection** → Updates selected_* variables  
4. **Analysis Execution** → Sets run_analysis_executed = True
5. **Results Display** → Uses stored results and selections

## 📝 Type Hints

### Common Types

```python
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np

# Dataset type
DataFrameType = pd.DataFrame

# Results type  
ResultsType = Dict[str, Union[float, str, Dict, List]]

# Constraints type
ConstraintsType = Dict[str, List[List[str]]]

# File upload type
from streamlit.runtime.uploaded_file_manager import UploadedFile
FileType = UploadedFile
```

### Function Signatures

```python
def calculate_ate(
    treatment: str,
    outcome: str, 
    confounders: Optional[List[str]] = None
) -> ResultsType:
    ...

def generate_domain_constraints(
    columns: List[str],
    domain_context: str,
    api_key: Optional[str] = None
) -> ConstraintsType:
    ...
```

---

*This API reference provides complete technical documentation for developers working with the Causal AI Platform.*
