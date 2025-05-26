# 🤖 Causal AI Platform

A comprehensive platform for discovering and analyzing causal relationships in your data using advanced AI and statistical methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage Guide](#detailed-usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🌟 Overview

The Causal AI Platform combines cutting-edge causal inference methods with AI-powered insights to help you:

- **Discover** causal relationships in your data
- **Quantify** treatment effects with statistical confidence
- **Simulate** policy interventions and their impacts
- **Get AI explanations** of your results in business-friendly language

### What Makes This Platform Unique?

1. **Multiple Fallback Methods**: Works even without advanced packages
2. **AI Integration**: Optional OpenAI integration for enhanced insights
3. **Robust Data Handling**: Handles messy CSV/Excel files automatically
4. **Interactive Visualizations**: Beautiful, interactive causal graphs
5. **Business-Focused**: Results explained in actionable business terms

## ✨ Features

### Core Capabilities

- **📁 Data Upload & Cleaning**
  - Support for Excel (.xlsx) and CSV files
  - Automatic data type detection and conversion
  - Robust handling of missing values and formatting issues
  - Smart column name cleaning

- **🔍 Causal Discovery**
  - LiNGAM-based causal structure learning
  - Correlation-based fallback methods
  - Domain constraint integration
  - Interactive causal graph visualization

- **📊 Causal Inference**
  - Average Treatment Effect (ATE) calculation
  - Multiple estimation methods (DoWhy + sklearn fallbacks)
  - Confidence intervals and significance testing
  - Robustness checks and sensitivity analysis

- **🧠 AI-Powered Features**
  - Domain-specific constraint generation
  - Business-friendly result explanations
  - Actionable recommendations
  - Context-aware insights

- **📈 Advanced Analytics**
  - Effect heterogeneity analysis
  - Policy intervention simulation
  - Variable relationship exploration
  - Export capabilities

### Supported Methods

| Method | Library | Fallback | Description |
|--------|---------|----------|-------------|
| DirectLiNGAM | lingam | Correlation-based | Causal structure discovery |
| DoWhy | dowhy | sklearn | Causal inference |
| GPT-3.5-turbo | openai | Manual analysis | AI explanations |

## 🏗️ Architecture

```
c:\Projects\streamlit\
├── 📁 models/                 # Core analysis models
│   ├── __init__.py
│   └── causal_analyzer.py     # Main CausalAnalyzer class
├── 📁 utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_cleaning.py       # Data preprocessing
│   ├── constraint_utils.py    # Constraint handling
│   └── fallback_methods.py    # Backup methods
├── 📁 calculations/           # Statistical calculations
│   ├── __init__.py
│   ├── causal_inference.py    # ATE calculations
│   ├── metrics.py             # Effect size & metrics
│   ├── relationship_analysis.py # Correlation analysis
│   └── advanced_analysis.py   # Heterogeneity & simulation
├── 📁 ai/                     # AI integration
│   ├── __init__.py
│   └── llm_integration.py     # OpenAI integration
├── 📁 ui/                     # UI components
│   ├── __init__.py
│   └── components.py          # Reusable UI elements
├── 📁 docs/                   # Documentation
│   ├── user_guide.md
│   ├── api_reference.md
│   └── examples/
├── streamlit_app.py           # Main application
└── README.md                  # This file
```

### Design Principles

- **Modular Architecture**: Clean separation of concerns
- **Graceful Degradation**: Works with or without optional packages
- **User-Centric**: Focus on ease of use and clear explanations
- **Extensible**: Easy to add new methods and features

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Streamlit
- Basic scientific computing packages

### Quick Install

```bash
# Clone or download the project
cd c:\Projects\streamlit

# Install required packages
pip install streamlit pandas numpy plotly networkx scikit-learn scipy

# Optional: Install advanced packages for full functionality
pip install lingam dowhy openai

# Run the application
streamlit run streamlit_app.py
```

### Package Dependencies

#### Required (Core Functionality)
```bash
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.20.0
plotly>=5.0.0
networkx>=2.8.0
scikit-learn>=1.1.0
scipy>=1.9.0
```

#### Optional (Enhanced Features)
```bash
lingam>=1.8.0          # Advanced causal discovery
dowhy>=0.9.0           # Robust causal inference
openai>=1.0.0          # AI-powered insights
```

## 🎯 Quick Start

### 1. Launch the Application

```bash
streamlit run streamlit_app.py
```

### 2. Upload Your Data

- Click "Browse files" and upload your Excel (.xlsx) or CSV file
- Ensure your data has:
  - At least 2 numeric columns
  - At least 10 rows
  - Clean column names (no special characters)

### 3. Follow the 5-Step Process

1. **📁 Data Upload** - Upload and preview your data
2. **🧠 Domain Constraints** - Optional AI-powered constraint generation
3. **🔍 Causal Discovery** - Discover causal relationships
4. **📊 Relationship Analysis** - Explore correlations
5. **🔬 Causal Inference** - Calculate treatment effects

### 4. Get Results

- View causal effect estimates
- See confidence intervals and p-values
- Get AI explanations (with API key)
- Export results for further analysis

## 📖 Detailed Usage Guide

### Data Requirements

#### Supported Formats
- **Excel (.xlsx)**: Recommended for best compatibility
- **CSV (.csv)**: Multiple encoding strategies attempted

#### Data Quality Tips
- Use numeric data for causal analysis
- Avoid special characters in column names
- Minimum 10 rows, 2 columns required
- Handle missing values beforehand when possible

#### Example Data Structure
```csv
Age,Income,Education,Health_Score,Exercise_Hours
25,50000,16,7.5,3
30,65000,18,8.2,5
35,75000,16,6.8,2
...
```

### Step-by-Step Workflow

#### Step 1: Data Upload & Cleaning

The platform automatically:
- Detects file encoding and format
- Cleans column names (removes quotes, special chars)
- Converts text to numeric where possible
- Handles missing values with median imputation
- Removes columns with >70% missing data

**What to expect:**
- Progress messages showing conversion attempts
- Summary of successful/failed conversions
- Final dataset preview with clean numeric data

#### Step 2: Domain Constraints (AI-Powered)

**Purpose**: Guide causal discovery with domain knowledge

**How to use:**
1. Describe your data context in the text area
2. Click "Generate Domain Constraints"
3. Review AI-suggested constraints
4. Constraints automatically applied to causal discovery

**Example prompt:**
```
This is customer behavior data where:
- Age and Income are demographic factors
- Marketing Spend influences Purchase Behavior
- Demographics may affect both Marketing effectiveness and Purchases
- Purchase Behavior cannot influence Age or Income
```

**What you get:**
- Forbidden edges (impossible causal relationships)
- Required edges (known causal relationships)
- Temporal ordering (cause must precede effect)
- Human-readable explanation

#### Step 3: Causal Discovery

**What it does:**
- Learns causal structure from data
- Creates directed acyclic graph (DAG)
- Uses LiNGAM or correlation-based methods

**How to interpret results:**
- **Nodes**: Your variables
- **Arrows**: Causal relationships (A → B means A causes B)
- **Weights**: Strength of causal relationship
- **Adjacency Matrix**: Numerical representation

#### Step 4: Variable Relationship Analysis

**Purpose**: Understand correlations before causal analysis

**What you get:**
- Correlation heatmap
- Strongest correlations list
- Suggested treatment-outcome pairs

**Use this to:**
- Identify potential causal relationships
- Choose treatment and outcome variables
- Spot data quality issues

#### Step 5: Causal Inference Analysis

**Core Process:**
1. Select treatment variable (cause)
2. Select outcome variable (effect)
3. Choose confounding variables (optional but recommended)
4. Run causal inference

**Results include:**
- **Causal Effect Estimate**: Quantified treatment effect
- **Confidence Intervals**: Statistical uncertainty
- **P-values**: Statistical significance
- **Effect Size Classification**: Small/Medium/Large
- **Business Interpretation**: What the numbers mean

### Advanced Features

#### Effect Heterogeneity Analysis

**Purpose**: Understand how treatment effects vary across subgroups

**How to use:**
1. Select a moderator variable
2. Click "Analyze Effect Heterogeneity"
3. Compare effects between high/low groups

**Example**: Does the effect of marketing spend on sales differ between young and old customers?

#### Policy Intervention Simulation

**Purpose**: Predict outcomes of proposed changes

**How to use:**
1. Specify intervention size (e.g., increase marketing by $1000)
2. Click "Simulate Policy Impact"
3. See predicted outcome changes

**Example**: "Increasing marketing spend by $1000 may increase sales by $2500 (5.2%)"

#### Robustness Analysis

**Purpose**: Test if results are stable under different assumptions

**What it tests:**
- Random common cause addition
- Data subset stability
- Assumption violations

**How to interpret:**
- ✅ Robust: Effect remains similar
- ⚠️ Sensitive: Effect changes significantly
- ❌ Failed: Test couldn't be performed

## 🔧 API Reference

### Core Classes

#### CausalAnalyzer

Main class for causal analysis pipeline.

```python
from models.causal_analyzer import CausalAnalyzer

analyzer = CausalAnalyzer()
```

**Methods:**

```python
# Load and clean data
success = analyzer.load_data(uploaded_file)

# Run causal discovery
success = analyzer.run_causal_discovery(constraints=None)

# Calculate treatment effects
results = analyzer.calculate_ate(treatment, outcome, confounders=[])

# Analyze relationships
relationships = analyzer.analyze_variable_relationships()

# Effect heterogeneity
hetero_results = analyzer.analyze_effect_heterogeneity(treatment, outcome, moderator)

# Policy simulation
sim_results = analyzer.simulate_policy_intervention(treatment, outcome, size)
```

### Utility Functions

#### Data Cleaning

```python
from utils.data_cleaning import clean_data

clean_df = clean_data(raw_df)
```

#### Constraint Utilities

```python
from utils.constraint_utils import convert_edge_constraints, adjacency_to_graph_string

# Convert edge list to matrix format
matrix = convert_edge_constraints(edge_list, columns)

# Convert adjacency matrix to DoWhy format
graph_string = adjacency_to_graph_string(adjacency_matrix, columns)
```

### AI Integration

#### LLM Functions

```python
from ai.llm_integration import generate_domain_constraints, explain_results_with_llm

# Generate constraints
constraints = generate_domain_constraints(columns, context, api_key)

# Explain results
explanation = explain_results_with_llm(results, treatment, outcome, api_key)
```

## ⚙️ Configuration

### Environment Variables

Create `.streamlit/secrets.toml`:

```toml
# OpenAI Configuration (optional)
OPENAI_API_KEY = "your-openai-api-key-here"

# Other configurations
DEFAULT_CORRELATION_THRESHOLD = 0.3
MAX_MISSING_PERCENTAGE = 70
MIN_ROWS_REQUIRED = 10
MIN_COLUMNS_REQUIRED = 2
```

### Package Configuration

#### With All Features (Recommended)

```bash
# Full installation
pip install lingam dowhy openai streamlit pandas numpy plotly networkx scikit-learn scipy
```

#### Minimal Installation (Basic Functionality)

```bash
# Core packages only
pip install streamlit pandas numpy plotly networkx scikit-learn scipy
```

### OpenAI API Setup

1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to sidebar in the app, OR
3. Add to `.streamlit/secrets.toml` file

**Cost considerations:**
- Uses GPT-3.5-turbo (cheaper than GPT-4)
- Typical cost: $0.01-0.05 per analysis
- Only charges when AI features are used

## 🔧 Troubleshooting

### Common Issues

#### Data Loading Problems

**Issue**: "Unable to parse CSV file"
**Solution**: 
- Try saving as Excel (.xlsx) format
- Check for inconsistent number of columns
- Remove special characters from data

**Issue**: "Not enough numeric columns"
**Solution**:
- Ensure at least 2 columns contain numbers
- Remove currency symbols ($, €)
- Use dots (.) for decimal separators

#### Causal Discovery Issues

**Issue**: "Error in causal discovery"
**Solution**:
- Platform automatically falls back to correlation-based methods
- Check data quality and size
- Ensure sufficient variation in variables

#### AI Features Not Working

**Issue**: "OpenAI not available"
**Solution**:
- Install openai package: `pip install openai`
- Restart the application

**Issue**: "API Key Error"
**Solution**:
- Verify API key is correct
- Check API key has sufficient credits
- Ensure key has access to GPT-3.5-turbo

### Performance Optimization

#### For Large Datasets
- Use Excel format when possible
- Pre-clean data to remove unnecessary columns
- Consider sampling for initial exploration

#### For Slow AI Responses
- AI features are optional
- Platform works fully without API key
- Consider upgrading OpenAI plan for faster responses

### Memory Issues

#### Large Files
- Platform automatically handles memory efficiently
- Consider splitting very large datasets
- Remove unnecessary columns before upload

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd streamlit

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run application
streamlit run streamlit_app.py
```

### Code Structure Guidelines

- **Models**: Core business logic
- **Utils**: Reusable utility functions
- **Calculations**: Statistical computations
- **AI**: External API integrations
- **UI**: User interface components

### Adding New Features

1. **New Calculation Method**: Add to `calculations/`
2. **New Data Source**: Extend `utils/data_cleaning.py`
3. **New AI Feature**: Add to `ai/llm_integration.py`
4. **New UI Component**: Add to `ui/components.py`

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/test_causal_analyzer.py
python -m pytest tests/test_data_cleaning.py
```

## 📚 Additional Resources

### Learning Materials

- **Causal Inference**: "The Book of Why" by Judea Pearl
- **Statistical Methods**: "Causal Inference: The Mixtape" by Scott Cunningham
- **Practical Applications**: "Causal Inference for the Brave and True"

### Related Tools

- **DoWhy**: Microsoft's causal inference library
- **LiNGAM**: Linear Non-Gaussian Acyclic Model for causal discovery
- **CausalML**: Uber's machine learning library for causal inference

### Academic References

- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Shimizu, S. et al. (2006). A Linear Non-Gaussian Acyclic Model for Causal Discovery
- Sharma, A. & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

1. Check this documentation first
2. Review the troubleshooting section
3. Check existing GitHub issues
4. Create a new issue with detailed information

---

**Happy Causal Analysis! 🚀**

*Remember: Correlation is not causation, but this platform helps you find the difference!*