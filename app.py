import streamlit as st
import torch
import numpy as np
import os
import cv2
from architecture import model_generator
from utils import save_matv73
from test import forward_ensemble

# Constants
MODEL_PATH = "./model_zoo/mst_plus_plus.pth"
METHOD = "mst_plus_plus"
ENSEMBLE_MODE = "mean"
VAR_NAME = "cube"

# Load model once
@st.cache_resource
def load_model():
    model = model_generator(METHOD, MODEL_PATH)
    model = model.to("cpu")
    model.eval()
    return model

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    return torch.from_numpy(rgb).float()

def postprocess_and_save(output_tensor, filename):
    result = output_tensor.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.clip(result, 0, 1)
    
    mat_path = os.path.join("temp_outputs", filename.replace(".jpg", ".mat"))
    if not os.path.exists("temp_outputs"):
        os.makedirs("temp_outputs")
    save_matv73(mat_path, VAR_NAME, result)
    return mat_path

# Streamlit UI
st.title("RGB to Hyperspectral Image Converter 🌈")

uploaded_file = st.file_uploader("Upload an RGB image (JPEG format)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    with st.spinner("Converting to hyperspectral..."):
        input_tensor = preprocess_image(uploaded_file).to("cpu")
        with torch.no_grad():
            result = forward_ensemble(input_tensor, model, ENSEMBLE_MODE)
        mat_path = postprocess_and_save(result, uploaded_file.name)
    
    st.success("Hyperspectral image created!")
    with open(mat_path, "rb") as f:
        st.download_button("Download Hyperspectral .mat File", f, file_name=os.path.basename(mat_path))
