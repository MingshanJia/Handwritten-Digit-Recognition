import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras as k
import numpy as np
import tensorflow as tf
import cv2
import base64

loaded_CNN = k.models.load_model('CNN_v2.h5')

def get_image_base64_str(file_path):
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_image}"

st.markdown(
    """
    <style>
    .logo {
        position: absolute;
        bottom: 10px;
        right: 10px;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the image using the st.markdown function and an HTML img tag
image_base64_str = get_image_base64_str("tafe_nsw_2.jpg")
st.markdown(f'<img src="{image_base64_str}" class="logo" width="100">', unsafe_allow_html=True)

st.title("Draw a Digit")
st.header("Draw a digit in the canvas below and the model will predict it.")

def invert_colors(image):
    return 255 - image

def crop_and_resize_digit(image, target_size=(28, 28), padding=2):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cv2.resize(image, target_size)

    # Find bounding box
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Crop and resize
    cropped_image = image[y:y + h, x:x + w]
    resized_image = cv2.resize(cropped_image, target_size)

    return resized_image

# Create a canvas for user input
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0.3)",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
)

# Add a button for prediction
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = crop_and_resize_digit(canvas_result.image_data.astype('uint8'))
        img = invert_colors(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_rescaled = img_gray / 255.0

        img_input = img_gray_rescaled.reshape(1, 28, 28, 1)
        prediction = loaded_CNN.predict(img_input)

        best_prediction = np.argmax(prediction)
        st.subheader(f"Predicted digit: {best_prediction}")

        top_2 = tf.math.top_k(prediction, k=2)
        classes = top_2.indices[0]
        probabilities = top_2.values[0]

        st.subheader("Top 2 Predictions with Probabilities:")
        for i in range(2):
            st.write(f"Prediction {i + 1}: class {classes[i]}, probability {probabilities[i] * 100:.2f}%")
    else:
        st.warning("Please draw a digit in the canvas.")