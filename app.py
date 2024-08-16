import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cnn_classification_model.h5')

# Set the title of the app
st.title("Bone Fracture Detection App")

# Add a file uploader to allow users to upload a photo
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    # Resize the image to 224x224 pixels as required by the model
    image = image.resize((224, 224))
    image = np.array(image)
    if len(image.shape) == 2:  # If grayscale, add an extra dimension
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)

    # Display the prediction
    st.write(f"Prediction: {predicted_class[0]}")
else:
    st.write("Please upload an X-ray image to get a prediction.")
