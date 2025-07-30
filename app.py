import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load models
pneumonia_model = load_model("chest_xray_model.h5")
brain_model = load_model("brain_tumor.h5")

# Constants
IMG_SIZE = 64

# Title
st.title("Medical Image Diagnosis")
st.write("Upload an image to detect **Pneumonia** or **Brain Tumor** using trained models.")

# Sidebar model selection
option = st.sidebar.selectbox(
    "Choose the model",
    ("Pneumonia Detection", "Brain Tumor Detection")
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Preprocessing function (both models expect RGB)
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')  # Always RGB
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Making prediction..."):
        processed_image = preprocess_image(image)

        if option == "Pneumonia Detection":
            prediction = pneumonia_model.predict(processed_image)[0][0]
            result = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia"
        else:
            prediction = brain_model.predict(processed_image)[0][0]
            result = "Brain Tumor Detected" if prediction > 0.5 else "No Brain Tumor"

        st.success(f"**Prediction:** {result}")
        st.info(f"Raw Prediction Score: {prediction:.4f}")
