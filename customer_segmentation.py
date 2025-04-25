# -*- coding: utf-8 -*-


import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Page title
st.title("🔍 K-Means Clustering App")
st.write("เลือก Dataset ด้านล่างเพื่อลองทำ Clustering และดูผลการกระจายกลุ่ม")

# Dataset selection
dataset_option = st.sidebar.selectbox("📂 เลือก Dataset", ["Iris Dataset", "Mall Customer Dataset"])

if dataset_option == "Iris Dataset":
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataset_name = "Iris Dataset"
    
elif dataset_option == "Mall Customer Dataset":
    try:
        df = pd.read_csv("Mall_Customers (2).csv")
    except:
        st.error("ไม่พบไฟล์ 'Mall_Customers (2).csv' กรุณาอัปโหลดหรือวางในโฟลเดอร์เดียวกับสคริปต์นี้")
        st.stop()
    
    dataset_name = "Mall Customer Dataset"
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Cluster number selection
k = st.sidebar.slider("เลือกจำนวนกลุ่ม (k)", min_value=2, max_value=10, value=5)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans modeling
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=50)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='o', label='Centroids')
ax.set_title(f"{dataset_name} - K={k} Clusters")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()
st.pyplot(fig)

# Display data
st.write("### 🧾 Clustered Data (แสดงเพียง 10 แถว)")
if dataset_option == "Iris Dataset":
    st.dataframe(X.assign(Cluster=labels).head(10))
else:
    st.dataframe(df.assign(Cluster=labels).head(10))

# Show Centroids
st.write("### 📌 Cluster Centers (scaled, PCA-reduced)")
st.dataframe(pd.DataFrame(centroids_pca, columns=["PCA1", "PCA2"]))
