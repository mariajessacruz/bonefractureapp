import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cnn_classification_model.h5')

# Set the title of the web app
st.title("Bone Fracture Detection")

# Add a file uploader to allow users to upload a photo
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# Define the class labels based on your training
class_labels = {
    0: "No fracture",
    1: "Fracture type A",
    2: "Fracture type B",
    3: "Fracture type C",
    4: "Fracture type D",
    5: "Fracture type E",
    6: "Fracture type F",
}

def preprocess_image(image):
    # Resize the image to 224x224 pixels as required by the model
    image = image.resize((224, 224))
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def extract_features(image):
    # Ensure the image is a TensorFlow tensor
    gray_image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Expand dimensions to [batch_size, height, width, channels]
    gray_image = tf.expand_dims(gray_image, axis=0)  # Shape: (1, 224, 224, 1)

    # Perform Sobel edge detection
    edges = tf.image.sobel_edges(gray_image)
    edges = tf.reduce_mean(edges, axis=-1).numpy().squeeze()

    # Flatten the edge image to simulate feature extraction
    hog_features = edges.flatten() / 255.0
    
    # Adjust the number of features to match the expected input shape
    hog_features = hog_features[:26244]  # Ensure the feature vector has 26244 elements
    
    return gray_image, edges, hog_features

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image for the main input
    processed_image = preprocess_image(image)
    
    # Extract additional features required by the model
    gray_image, edges, hog_features = extract_features(processed_image)
    
    # Expand dimensions to fit model's expected input shapes
    processed_image = np.expand_dims(processed_image, axis=0)  # Shape: (1, 224, 224, 1)
    hog_features = np.expand_dims(hog_features, axis=0)  # Shape: (1, 26244)
    edges = np.expand_dims(edges, axis=0)  # Shape: (1, 224, 224, 1)

    # Perform prediction using all three inputs
    prediction = model.predict([processed_image, hog_features, gray_image])
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the prediction with class label
    st.write(f"Prediction: {class_labels[predicted_class]}")
else:
    st.write("Please upload an X-ray image to get a prediction.")
