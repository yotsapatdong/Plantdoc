import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.title("🌿 ระบบตรวจจับโรคพืช")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("อัปโหลดภาพใบพืช", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="ภาพต้นฉบับ", use_column_width=True)

    # แปลงภาพ
    img_cv = np.array(img)
    results = model.predict(img_cv)

    # แสดงผลลัพธ์
    res_plotted = results[0].plot()  # วาดกล่อง
    st.image(res_plotted, caption="ผลตรวจจับ", use_column_width=True)

    # รายงาน class ที่ตรวจเจอ
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    st.write("🩺 ตรวจพบ:", ", ".join(labels) if labels else "ไม่พบโรคพืช")
