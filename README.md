# 🔍 Causal Discovery App

A Streamlit-based web application for exploring causal relationships between variables using the DirectLiNGAM algorithm. This prototype demonstrates how to build interactive data science applications for business presentations and client demos.

## 📖 Description

This application uses **DirectLiNGAM (Direct Linear Non-Gaussian Acyclic Model)** to discover causal relationships in data. The app features:

- **Interactive Data Visualization**: View simulated datasets with known causal structures
- **Real-time Causal Discovery**: Run the DirectLiNGAM algorithm with a single click
- **Visual Causal Graphs**: Network diagrams showing discovered relationships
- **Adjacency Matrix Display**: Numerical representation of causal strengths
- **Insights Dashboard**: Summary statistics about discovered relationships

### 🧠 What is Causal Discovery?

Causal discovery aims to identify cause-and-effect relationships from observational data. Unlike correlation, causation implies that changes in one variable directly influence another. This is crucial for:

- **Business Decision Making**: Understanding what drives key metrics
- **Scientific Research**: Identifying fundamental relationships
- **Policy Making**: Predicting intervention effects
- **Machine Learning**: Building more robust predictive models

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for version control)
- GitHub account (for hosting)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd causal-discovery-app
   ```

2. **Create virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically reload when you make changes

## 📦 Dependencies

```txt
streamlit==1.28.0      # Web app framework
pandas==2.0.3          # Data manipulation
matplotlib==3.7.2      # Plotting and visualization
networkx==3.1          # Graph/network analysis
lingam==1.8.2          # Causal discovery algorithms
numpy==1.24.3          # Numerical computing
```

### Why These Libraries?

- **Streamlit**: Rapidly build data apps with minimal code
- **LiNGAM**: State-of-the-art causal discovery implementation
- **NetworkX**: Professional graph visualization and analysis
- **Pandas**: Industry-standard data manipulation
- **Matplotlib**: Reliable, customizable plotting

## 🏗️ Project Structure

```
causal-discovery-app/
│
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── .gitignore              # Git ignore patterns
│
└── (generated files)
    ├── .streamlit/         # Streamlit configuration
    └── __pycache__/        # Python cache files
```

## 🔧 Development Guide

### Understanding the Code

The application consists of several key components:

#### 1. Data Generation (`generate_simulated_data()`)
```python
# Creates synthetic data with known causal structure:
# X1 → X2 → X3 (chain)
# X1 → X4 (direct effect)
```

#### 2. Causal Discovery (`DirectLiNGAM`)
```python
# Fits the model and discovers causal relationships
model = DirectLiNGAM()
model.fit(df)
adjacency_matrix = model.adjacency_matrix_
```

#### 3. Visualization (`NetworkX` + `Matplotlib`)
```python
# Creates interactive network graphs
G = nx.DiGraph(adjacency_matrix)
nx.draw(G, pos, with_labels=True, ...)
```

### Customization Options

**Adding New Algorithms:**
```python
# Add other causal discovery methods
from lingam import ICALiNGAM, VARLiNGAM

# Create algorithm selector
algorithm = st.selectbox("Choose Algorithm", 
                        ["DirectLiNGAM", "ICALiNGAM", "VARLiNGAM"])
```

**Custom Data Sources:**
```python
# Add file upload functionality
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
```

**Enhanced Visualizations:**
```python
# Add interactive plots with Plotly
import plotly.graph_objects as go
fig = go.Figure(data=go.Heatmap(z=adjacency_matrix))
st.plotly_chart(fig)
```

## 🌐 Deployment & Hosting

### Option 1: Streamlit Community Cloud (Recommended for Startups)

**Advantages:**
- ✅ Free tier with no time limits
- ✅ Professional URLs (`yourapp.streamlit.app`)
- ✅ Auto-deployment from GitHub
- ✅ Zero DevOps overhead
- ✅ Perfect for client demos

**Steps:**
1. Push code to GitHub (public repository)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Set main file: `streamlit_app.py`
6. Deploy!

**Your app will be live at:** `https://your-repo-name.streamlit.app`

### Option 2: Render (Alternative)

**Steps:**
1. Connect GitHub at [render.com](https://render.com)
2. Choose "Web Service"
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run streamlit_app.py --server.port $PORT`

### Option 3: Railway (Quick Deploy)

**Steps:**
1. Visit [railway.app](https://railway.app)
2. Deploy from GitHub
3. Automatic Python detection

## 📊 Business Use Cases

### For Startups & Companies

**Marketing Analytics:**
- Discover what drives customer acquisition
- Understand conversion funnel causality
- Optimize marketing spend allocation

**Operations Research:**
- Identify bottlenecks in processes
- Understand quality factor relationships
- Optimize supply chain decisions

**Financial Analysis:**
- Understand revenue drivers
- Risk factor identification
- Investment decision support

**Healthcare & Pharma:**
- Treatment effectiveness analysis
- Side effect identification
- Patient outcome prediction

## 🎯 Demo Tips for Client Presentations

### 1. Prepare Your Narrative
```markdown
"This prototype demonstrates how AI can automatically discover 
cause-and-effect relationships in your data, enabling 
data-driven decision making."
```

### 2. Highlight Key Features
- **One-click analysis**: No technical expertise required
- **Visual results**: Easy-to-understand network diagrams
- **Scalable**: Works with any tabular dataset
- **Fast**: Results in seconds, not hours

### 3. Business Value Proposition
- **Reduce guesswork**: Make decisions based on causal evidence
- **Increase ROI**: Focus interventions on true drivers
- **Risk mitigation**: Understand unintended consequences
- **Competitive advantage**: Data-driven insights

## 🔄 Version Control & Collaboration

### Git Workflow

**Initial Setup:**
```bash
git init
git add .
git commit -m "Initial commit: Causal Discovery Streamlit App"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

**Development Workflow:**
```bash
# Create feature branch
git checkout -b feature/new-algorithm

# Make changes and commit
git add .
git commit -m "Add new causal discovery algorithm"

# Push and create pull request
git push origin feature/new-algorithm
```

**Deployment:**
- Any push to `main` branch automatically deploys to Streamlit Cloud
- Test locally before pushing to production

## 🧪 Testing & Validation

### Validation Strategy

The app uses simulated data with **known causal structure**:
- X1 → X2 → X3 (coefficient: 0.8 → 0.6)
- X1 → X4 (coefficient: 0.7)

**Expected Results:**
- DirectLiNGAM should recover these relationships
- Adjacency matrix should show non-zero values for true edges
- False positive rate should be low

### Adding Unit Tests

```python
# test_app.py
import pytest
import pandas as pd
from streamlit_app import generate_simulated_data

def test_data_generation():
    df = generate_simulated_data()
    assert df.shape == (1000, 4)
    assert list(df.columns) == ['Variable_1', 'Variable_2', 'Variable_3', 'Variable_4']
```

## 📚 Additional Resources

### Causal Discovery Learning
- [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
- [The Book of Why - Judea Pearl](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/0465097618)
- [Causal Inference in Statistics - Pearl, Glymour, Jewell](https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186847)

### Streamlit Development
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)

### DirectLiNGAM Algorithm
- [Original Paper](https://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf)
- [LiNGAM Package Documentation](https://lingam.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact & Support

- **GitHub Issues**: For bug reports and feature requests
- **Email**: your-email@company.com
- **LinkedIn**: Your professional profile

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Streamlit for rapid prototyping and client demonstrations.**