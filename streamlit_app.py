import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from lingam import DirectLiNGAM

st.set_page_config(page_title="Causal Discovery Prototype", layout="wide")

st.title("🔍 Causal Discovery Prototype")
st.markdown("Upload your dataset to explore causal relationships between variables.")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    if st.button("Run Causal Discovery"):
        st.subheader("📈 Causal Graph")
        model = DirectLiNGAM()
        model.fit(df)

        adjacency_matrix = model.adjacency_matrix_
        G = nx.DiGraph(adjacency_matrix)
        nx.relabel_nodes(G, dict(enumerate(df.columns)), copy=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, edge_color="gray", arrows=True, ax=ax)
        st.pyplot(fig)

        st.success("Causal discovery completed successfully.")
        st.markdown("### 📉 Adjacency Matrix")
        st.dataframe(pd.DataFrame(adjacency_matrix, index=df.columns, columns=df.columns))
