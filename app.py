import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import os
from PIL import Image

st.title("🌿 ระบบตรวจจับโรคพืช")

# -----------------------------
# โหลดโมเดลจาก Google Drive
# -----------------------------
model_path = "best.pt"

if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/file/d/1bgYi59vfzhvNZ9aL1-_Bi6pH2NfOyCbh/view?usp=drive_link"  # 👈 YOUR_FILE_ID เป็นของคุณ
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    st.success("✅ Model downloaded successfully!")
    
model = YOLO("yolo11n.pt")

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
