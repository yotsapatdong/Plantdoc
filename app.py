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
    url = "https://drive.google.com/uc?id=1bgYi59vfzhvNZ9aL1-_Bi6pH2NfOyCbh"  # 👈 YOUR_FILE_ID เป็นของคุณ
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    st.success("✅ Model downloaded successfully!")
    
model = YOLO(model_path)

uploaded_file = st.file_uploader("อัปโหลดภาพใบพืช", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🔹 เปิดภาพและแปลงให้เป็น RGB เสมอ (ป้องกัน error input)
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="ภาพต้นฉบับ", use_column_width=True)

    # 🔹 แปลงภาพเป็น NumPy Array
    img_cv = np.array(img)
    st.write("ขนาดภาพ:", img_cv.shape)  # debug เล็กน้อย จะได้รู้ว่ารูปเป็น (H, W, 3)

    # 🔹 ตรวจจับด้วย YOLO
    st.write("🧠 กำลังตรวจจับโรคพืช...")
    results = model.predict(source=img_cv, conf=0.5)

    # 🔹 แสดงผลลัพธ์
    res_plotted = results[0].plot()  # วาดกรอบผลลัพธ์
    st.image(res_plotted, caption="🔍 ผลตรวจจับ", use_column_width=True)

    # 🔹 รายงาน class ที่ตรวจเจอ
    if len(results[0].boxes) > 0:
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
        st.write("🩺 ตรวจพบ:", ", ".join(labels))
    else:
        st.write("✅ ไม่พบโรคพืชในภาพนี้")

