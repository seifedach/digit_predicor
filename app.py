# app.py

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import base64

# Load model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image, invert, resize_dim):
    image = ImageOps.grayscale(image)
    if invert:
        image = ImageOps.invert(image)
    image = image.resize((resize_dim, resize_dim))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, resize_dim, resize_dim, 1)
    return image

# Function to download the model
def get_model_download_link(model, filename="mnist_cnn_model.h5"):
    model.save(filename)
    with open(filename, "rb") as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'<a href="data:file/tfmodel;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

# Define the Streamlit app
def main():
    st.title('Numbers Image Recognition')
    st.markdown('This app recognizes handwritten digits using a CNN model.')

    st.sidebar.header('Upload and Preprocess Image')
    uploaded_file = st.sidebar.file_uploader("Choose a digit image...", type=["jpg", "jpeg", "png"])
    
    st.sidebar.header('Preprocess Options')
    invert = st.sidebar.checkbox('Invert Colors', value=True)
    resize_dim = st.sidebar.slider('Resize Dimension', min_value=28, max_value=100, value=28, step=1)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.sidebar.button('Predict'):
                processed_image = preprocess_image(image, invert, resize_dim)
                prediction = model.predict(processed_image)
                confidence = np.max(prediction)
                predicted_digit = np.argmax(prediction)
                st.write(f'**Predicted Digit:** {predicted_digit}')
                st.write(f'**Confidence:** {confidence:.2f}')
                
                st.markdown(get_model_download_link(model), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
