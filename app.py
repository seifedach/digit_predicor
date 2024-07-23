# app.py

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Define the Streamlit app
def main():
    st.title('Numbers Image Recognition')
    st.markdown('This app recognizes handwritten digits using a CNN model.')

    st.sidebar.header('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose a digit image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.sidebar.button('Predict'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                st.write(f'Predicted Digit: {np.argmax(prediction)}')
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
