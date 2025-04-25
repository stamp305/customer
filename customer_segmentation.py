

import os
os.environ["OMP_NUM_THREADS"] = "1"  # ป้องกัน warning จาก sklearn บน Windows

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("🛍️ Mall Customers Segmentation (K-Means Clustering)")

# Load dataset
df = pd.read_csv("Mall_Customers (2).csv")

st.write("### 🔍 Raw Dataset Preview")
st.dataframe(df.head())

# Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Add cluster label to original data
df["Cluster"] = labels

# Plot clustering
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set1', s=50)
ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.7, marker='o', label='Centroids')
ax.set_xlabel("Annual Income (scaled)")
ax.set_ylabel("Spending Score (scaled)")
ax.set_title("Customer Segmentation with K=5 Clusters")
ax.legend()

st.pyplot(fig)

# Show data with cluster
st.write("### 🧾 Segmented Customer Data (with Cluster)")
st.dataframe(df[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head(10))

# Show centroids
st.write("### 📌 Cluster Centers (Centroids)")
st.write(pd.DataFrame(centroids, columns=["Income (scaled)", "Score (scaled)"]))
