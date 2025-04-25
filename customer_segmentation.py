# -*- coding: utf-8 -*-
"""
Created on April 25, 2025
Project: Customer Segmentation & Prediction App
@author: YourName
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

# Title
st.title("📊 Customer Segmentation & Prediction App")

# Load Data
data = pd.read_csv("Mall_Customers (2).csv")

st.subheader("🧾 Raw Dataset")
st.dataframe(data.head())

# Encode Gender
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Feature Selection
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Clustering
st.subheader("🧠 K-Means Clustering")
k = 5  # fixed number of clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# PCA for 2D Visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

fig1, ax1 = plt.subplots()
for cluster in range(k):
    cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
    ax1.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
# Plot centroids in red
centroids_2d = pca.transform(kmeans.cluster_centers_)
ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', s=200, marker='o', label='Centroids')
ax1.set_title("Clusters (2D PCA Projection)")
ax1.legend()
st.pyplot(fig1)

# Classification
st.subheader("🔍 Classification with Random Forest")
data['High Spender'] = data['Spending Score (1-100)'].apply(lambda x: 1 if x > 50 else 0)
X_cls = X.drop('Spending Score (1-100)', axis=1)
y_cls = data['High Spender']

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ Accuracy: {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# User prediction
st.subheader("🎯 Predict Spending Class")
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 15, 70, 30)
income = st.slider("Annual Income (k$)", 10, 150, 50)

# Predict user input
user_input = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],
    "Age": [age],
    "Annual Income (k$)": [income]
})

user_pred = clf.predict(user_input)
st.write("🧠 Prediction: ", "High Spender" if user_pred[0] == 1 else "Low Spender")
