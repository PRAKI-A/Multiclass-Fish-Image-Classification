# app/streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load your best model (change filename if needed)
MODEL_PATH = 'models/ResNet50.h5'  # or cnn_model.h5 or best performing one
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match training order)
CLASS_NAMES = list(model.class_names) if hasattr(model, 'class_names') else [
    'class1', 'class2', 'class3', 'class4', 'class5'  # Replace with actual class names
]

# Set image size
IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array /= 255.0  # Normalize
    return img_array

st.title("🐟 Fish Image Classifier")
st.markdown("Upload an image of a fish and get its predicted category!")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner('Classifying...'):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)[0]

            predicted_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(predictions))

            st.success(f"🎯 Predicted: **{predicted_class}**")
            st.info(f"📈 Confidence: {confidence*100:.2f}%")
