import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Brain Tumor Detection AI", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload MRI image → AI predicts tumor presence")

# ---------- CHECK MODEL ----------
if not os.path.exists("yolo_model.pt"):
    st.error("❌ yolo_model.pt not found in project folder")
    st.stop()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolo_model.pt")

model = load_model()

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    with st.spinner("Analyzing MRI..."):
        results = model(img_np)
        probs = results[0].probs

        class_id = probs.top1
        confidence = float(probs.top1conf)
        label = results[0].names[class_id]

    st.subheader("Prediction Result")
    st.write("Class:", label.upper())
    st.write("Confidence:", round(confidence, 4))

    if label.lower() == "yes":
        st.error("⚠ Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

    st.progress(int(confidence * 100))