import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# โหลดข้อมูล
data = pd.read_csv("Mall_Customers (2).csv")

# เลือก Features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]  # ใช้ 'data' แทน 'df'

# การปรับมาตรฐานข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# โหลดโมเดลที่บันทึกไว้
loaded_model = pickle.load(open('kmeans_model.pkl', 'rb'))

# ทำนายกลุ่มลูกค้า
y_kmeans = loaded_model.predict(X_scaled)

# แสดงผล
st.title("Customer Segmentation with KMeans")
st.write("แสดงผลการแบ่งกลุ่มลูกค้าด้วย KMeans Clustering")

# แสดงข้อมูล
st.write(data.head())  # ใช้ 'data' แทน 'df'

# แสดงกราฟ
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=100)
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
ax.set_title("Customer Segments")
ax.set_xlabel("Annual Income (scaled)")
ax.set_ylabel("Spending Score (scaled)")
ax.legend()

# แสดงกราฟใน Streamlit
st.pyplot(fig)
