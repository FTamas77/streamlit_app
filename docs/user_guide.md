# 📖 User Guide: Causal AI Platform

## 🎯 Getting Started

### What is Causal Analysis?

Causal analysis goes beyond correlation to answer questions like:
- **Does** increasing marketing spend **cause** higher sales?
- **What happens** if we change our pricing strategy?
- **Which factors** actually drive customer satisfaction?

Traditional analysis shows correlation (A and B move together), but causal analysis reveals causation (A actually influences B).

### When to Use This Platform

✅ **Good Use Cases:**
- Business decision making
- Policy impact assessment
- Understanding driver-outcome relationships
- A/B test result analysis
- Marketing attribution

❌ **Not Suitable For:**
- Pure prediction tasks
- Time series forecasting
- Small datasets (<10 rows)
- Purely categorical data

## 📊 Data Preparation Guide

### Ideal Data Structure

Your data should be structured with:
- **Rows**: Individual observations/units
- **Columns**: Variables/features
- **Values**: Numeric measurements

**Example: Marketing Analysis**
```csv
Customer_ID,Age,Income,Marketing_Spend,Website_Visits,Purchase_Amount
1,25,50000,120,15,250
2,30,65000,80,12,180
3,35,75000,200,25,450
```

### Data Quality Checklist

#### ✅ Essential Requirements
- [ ] At least 10 rows of data
- [ ] At least 2 numeric columns
- [ ] Clean column names (no special characters)
- [ ] Consistent data types within columns

#### 🎯 Recommended Practices
- [ ] Remove obvious outliers beforehand
- [ ] Ensure sufficient variation in all variables
- [ ] Include potential confounding variables
- [ ] Document data collection process

#### 🚨 Common Issues to Avoid
- [ ] Mixed data types in same column
- [ ] Too many missing values (>70% per column)
- [ ] Identical values in entire columns
- [ ] Special characters in numeric data

### Data Cleaning Tips

**Before Upload:**
```excel
# Good column names
Age, Income, Marketing_Spend, Sales

# Avoid
Age (years), Income ($), Marketing Spend, Sales!!!
```

**Handling Missing Values:**
- Platform automatically fills with median values
- Consider removing rows with too many missing values
- Document why data might be missing

**Numeric Formatting:**
```
# Good
1234.56, 1000, 0.75

# Platform can handle
"1,234.56", "$1,000", "75%"

# Problematic
"One thousand", "N/A", "TBD"
```

## 🔍 Step-by-Step Walkthrough

### Step 1: Data Upload

1. **Click "Browse files"**
2. **Select your Excel (.xlsx) or CSV file**
3. **Wait for automatic processing**

**What happens automatically:**
- File format detection
- Column name cleaning
- Data type conversion
- Missing value handling

**Success indicators:**
- ✅ Green "Data loaded successfully!" message
- Data preview table appears
- Metrics show rows/columns/missing values

**If upload fails:**
- Try Excel format instead of CSV
- Check file isn't corrupted
- Ensure file has data (not just headers)

### Step 2: Domain Constraints (Optional)

**Purpose**: Help the AI understand your business context

**How to write effective prompts:**

```
# Good prompt
This is e-commerce customer data where:
- Age and Income are customer demographics
- Marketing Spend is our advertising investment per customer
- Website Visits shows engagement
- Purchase Amount is the business outcome
- Demographics influence purchasing power
- Marketing drives both visits and purchases
- Purchases cannot influence age or income

# Too vague
This is business data about customers and sales.
```

**AI Output Example:**
```json
{
  "forbidden_edges": [
    ["Purchase_Amount", "Age"],
    ["Website_Visits", "Income"]
  ],
  "required_edges": [
    ["Marketing_Spend", "Website_Visits"],
    ["Marketing_Spend", "Purchase_Amount"]
  ],
  "temporal_order": ["Age", "Income", "Marketing_Spend", "Website_Visits", "Purchase_Amount"],
  "explanation": "Demographics come first, marketing activities follow, then engagement and outcomes"
}
```

### Step 3: Causal Discovery

**What it does**: Learns the causal structure from your data

**Click "Run Causal Discovery"** and wait for results.

**Interpreting the Causal Graph:**
- **Nodes** = Your variables
- **Arrows** = Causal relationships (A → B means A causes B)
- **No arrow** = No direct causal relationship found

**Example Interpretation:**
```
Marketing_Spend → Website_Visits → Purchase_Amount
             ↘                 ↗
               Income
```
This suggests:
- Marketing directly increases website visits
- Marketing indirectly affects purchases through visits
- Income also influences purchases
- Age might not have direct causal effects

### Step 4: Variable Relationships

**Purpose**: Explore correlations before causal analysis

**Use the correlation heatmap to:**
- Identify strongly correlated variables
- Spot potential data quality issues
- Choose treatment/outcome pairs

**Color coding:**
- 🔴 Red = Strong negative correlation
- ⚪ White = No correlation  
- 🔵 Blue = Strong positive correlation

**Look for:**
- Strong correlations (>0.5) as potential causal relationships
- Unexpected correlations (data quality issues)
- Variables that correlate with many others (potential confounders)

### Step 5: Causal Inference

**This is the main analysis!**

#### Selecting Variables

**Treatment Variable (Cause):**
- The factor you can control or change
- Examples: Marketing spend, price, training hours

**Outcome Variable (Effect):**
- The result you want to understand
- Examples: Sales, satisfaction, performance

**Confounding Variables:**
- Factors that influence both treatment and outcome
- Critical for accurate results!
- Examples: Customer demographics, market conditions

#### Understanding Results

**Causal Effect Estimate:**
```
Estimate: 0.0234
Interpretation: "A one-unit increase in Marketing_Spend increases Purchase_Amount by 0.0234 units on average"
```

**Confidence Intervals:**
```
95% CI: [0.0123, 0.0345]
Meaning: We're 95% confident the true effect is between 0.0123 and 0.0345
```

**P-values:**
```
P-value: 0.003
Meaning: Very strong evidence of causal effect (p < 0.05)
```

**Effect Size Classification:**
- **Very Small** (<0.01): Minimal practical impact
- **Small** (0.01-0.1): Noticeable but modest effect
- **Medium** (0.1-0.5): Substantial effect
- **Large** (>0.5): Major impact

## 🎯 Real-World Examples

### Example 1: Marketing Effectiveness

**Business Question**: Does our digital marketing spend actually drive sales?

**Data Setup:**
```csv
Customer_ID,Age,Income,Digital_Marketing,Store_Visits,Purchase_Amount
1,25,45000,50,3,120
2,34,62000,100,5,280
3,28,51000,75,4,190
```

**Analysis Steps:**
1. **Treatment**: Digital_Marketing
2. **Outcome**: Purchase_Amount  
3. **Confounders**: Age, Income (affect both marketing allocation and purchasing power)

**Sample Results:**
```
Causal Effect: 2.3
Interpretation: Each $1 in digital marketing increases purchases by $2.30
Confidence Interval: [1.8, 2.8]
Recommendation: Strong ROI - consider increasing marketing spend
```

### Example 2: Employee Training Impact

**Business Question**: Does training actually improve performance?

**Data Setup:**
```csv
Employee_ID,Years_Experience,Department,Training_Hours,Performance_Score
1,2,Sales,40,7.5
2,5,Sales,20,8.2
3,1,Marketing,60,6.8
```

**Analysis Steps:**
1. **Treatment**: Training_Hours
2. **Outcome**: Performance_Score
3. **Confounders**: Years_Experience, Department

**Sample Results:**
```
Causal Effect: 0.05
Interpretation: Each hour of training increases performance by 0.05 points
P-value: 0.12 (not significant)
Recommendation: Training effect not statistically significant - investigate training quality
```

### Example 3: Pricing Strategy

**Business Question**: How does price affect demand?

**Data Setup:**
```csv
Product_ID,Category,Brand_Strength,Price,Marketing_Support,Units_Sold
1,Electronics,8,299,1000,150
2,Electronics,6,199,500,280
3,Home,7,99,300,420
```

**Analysis Steps:**
1. **Treatment**: Price
2. **Outcome**: Units_Sold
3. **Confounders**: Category, Brand_Strength, Marketing_Support

**Sample Results:**
```
Causal Effect: -0.8
Interpretation: Each $1 price increase reduces sales by 0.8 units
Effect Size: Medium
Recommendation: Consider price optimization - significant demand elasticity
```

## 🔧 Advanced Features Guide

### Effect Heterogeneity Analysis

**Purpose**: Understand if treatment effects vary across subgroups

**When to use:**
- Effects might differ by customer segment
- Policy impacts might vary by region
- Treatment effectiveness varies by context

**Example:**
```
Treatment: Marketing_Spend
Outcome: Sales
Moderator: Customer_Age

Results:
- Young customers: Marketing effect = 3.2
- Older customers: Marketing effect = 1.8
- Difference: 1.4

Interpretation: Marketing is more effective for younger customers
```

### Policy Intervention Simulation

**Purpose**: Predict outcomes before implementing changes

**Use cases:**
- Budget planning
- Policy impact assessment
- Risk evaluation

**Example:**
```
Current marketing spend: $1000
Proposed increase: +$500
Predicted sales increase: +$1150 (15.2%)
ROI: 230%

Recommendation: Intervention likely profitable
```

### Robustness Analysis

**Purpose**: Test if results are reliable

**What it checks:**
- **Random common cause**: Adds random confounders
- **Data subset**: Uses partial data
- **Placebo treatment**: Tests with fake treatment

**Interpreting results:**
- ✅ **Robust**: Effect remains similar across tests
- ⚠️ **Sensitive**: Effect changes significantly
- ❌ **Failed**: Cannot determine robustness

## 🤖 AI Features Guide

### Domain Constraints Generation

**Best practices for prompts:**

**Be specific about:**
- Industry/domain context
- Temporal relationships
- Business logic
- Known impossibilities

**Example prompts:**

**E-commerce:**
```
This is e-commerce data where customers browse, add to cart, and purchase.
Customer demographics (age, income) are fixed characteristics.
Marketing campaigns influence browsing behavior.
Browsing leads to cart additions, which lead to purchases.
Purchase behavior cannot change demographics.
Seasonal factors affect all behavior.
```

**Healthcare:**
```
This is patient treatment data from a clinical trial.
Patient characteristics (age, baseline health) are pre-treatment.
Treatment assignment was randomized.
Treatment affects intermediate outcomes (symptoms, lab values).
Final outcomes (recovery, satisfaction) depend on intermediate outcomes.
Recovery cannot influence initial characteristics.
```

### AI Results Explanation

**What you get:**
- Business-friendly interpretation
- Actionable recommendations
- Reliability assessment
- Implementation guidance

**Example AI explanation:**
```
Your analysis shows a strong causal relationship between marketing spend and sales revenue. 

Key Findings:
- Each $1 invested in marketing generates $2.30 in additional sales
- This effect is statistically significant (p < 0.05)
- The confidence interval suggests the true effect is between $1.80-$2.80

Business Implications:
- Current marketing is highly profitable
- Consider increasing marketing budget
- Monitor for diminishing returns at higher spend levels

Reliability:
- Strong statistical evidence
- Consider running controlled experiment to validate
- Monitor key metrics if implementing changes

Recommendations:
1. Gradually increase marketing spend by 20-30%
2. Track incremental sales closely
3. Test different marketing channels for optimization
```

## ⚠️ Interpretation Guidelines

### Statistical Significance

**P-value < 0.05**: Strong evidence of causal effect
**P-value 0.05-0.10**: Moderate evidence, consider practical significance
**P-value > 0.10**: Weak evidence, investigate further

### Effect Sizes

**Consider both statistical and practical significance:**
- A small but significant effect might not be worth acting on
- A large effect with moderate p-value might still be important
- Context matters more than statistical rules

### Confidence Intervals

**Wide intervals**: Less precision, need more data
**Narrow intervals**: More precise estimates
**Intervals crossing zero**: Effect might not exist

### Common Pitfalls

❌ **Don't assume causation from correlation**
❌ **Don't ignore confounding variables**
❌ **Don't over-interpret small effects**
❌ **Don't ignore practical significance**

✅ **Do consider alternative explanations**
✅ **Do validate with experiments when possible**
✅ **Do consider business context**
✅ **Do monitor implementation results**

## 📈 Best Practices

### Data Collection
- Plan for causal analysis from the start
- Include potential confounders
- Ensure sufficient sample size
- Document data collection process

### Analysis
- Start with domain knowledge
- Use multiple methods when possible
- Check robustness of results
- Consider alternative explanations

### Implementation
- Start with small tests
- Monitor key metrics
- Be prepared to adjust
- Document learnings

### Communication
- Focus on business implications
- Acknowledge limitations
- Provide actionable recommendations
- Use AI explanations for clarity

---

*Remember: Causal analysis is powerful but requires careful interpretation. When in doubt, validate findings with controlled experiments!*
