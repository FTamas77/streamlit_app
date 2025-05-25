import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from lingam import DirectLiNGAM
import numpy as np

st.set_page_config(page_title="Causal Discovery Prototype", layout="wide")

st.title("🔍 Causal Discovery Prototype")
st.markdown("Explore causal relationships between variables using simulated data.")

# Generate simulated data
@st.cache_data
def generate_simulated_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Create causal relationships: X1 -> X2 -> X3, X1 -> X4
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.8 * X1 + np.random.normal(0, 0.5, n_samples)
    X3 = 0.6 * X2 + np.random.normal(0, 0.3, n_samples)
    X4 = 0.7 * X1 + np.random.normal(0, 0.4, n_samples)
    
    df = pd.DataFrame({
        'Variable_1': X1,
        'Variable_2': X2,
        'Variable_3': X3,
        'Variable_4': X4
    })
    return df

# Generate and display simulated data
df = generate_simulated_data()
st.subheader("📊 Simulated Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Samples", df.shape[0])
with col2:
    st.metric("Number of Variables", df.shape[1])

if st.button("Run Causal Discovery", type="primary"):
    with st.spinner("Running causal discovery algorithm..."):
        st.subheader("📈 Causal Graph")
        model = DirectLiNGAM()
        model.fit(df)

        adjacency_matrix = model.adjacency_matrix_
        G = nx.DiGraph(adjacency_matrix)
        nx.relabel_nodes(G, dict(enumerate(df.columns)), copy=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="skyblue", 
                node_size=3000, edge_color="gray", arrows=True, 
                font_size=10, font_weight="bold", ax=ax)
        ax.set_title("Discovered Causal Graph", fontsize=14, fontweight="bold")
        st.pyplot(fig)

        st.success("Causal discovery completed successfully! ✅")
        
        st.subheader("📉 Adjacency Matrix")
        adj_df = pd.DataFrame(adjacency_matrix, index=df.columns, columns=df.columns)
        st.dataframe(adj_df, use_container_width=True)
        
        # Add some insights
        st.subheader("📝 Insights")
        n_edges = np.sum(np.abs(adjacency_matrix) > 0.1)
        st.info(f"The algorithm discovered {n_edges} causal relationships with strength > 0.1")