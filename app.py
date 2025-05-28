import streamlit as st
import numpy as np
import cv2
import os
from keras.models import load_model
from PIL import Image

# Set title
st.title("Facial Emotion Detection")

# Try loading the model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('custom_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.error("No face detected in the image.")
        else:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            if model:
                prediction_idx = np.argmax(model.predict(face))
                prediction = emotion_labels[prediction_idx]
                st.success(f"Predicted Emotion: **{prediction}**")

                # Draw bounding box and show image
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(img_array, caption="Detected Face", use_column_width=True)
            else:
                st.error("Model is not loaded. Please upload the model file.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
