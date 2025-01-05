from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Function to classify the fruit
def classify_fruit(img):
    np.set_printoptions(suppress=True)  # Disable scientific notation

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create input array for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Convert image to RGB and resize
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Streamlit App Configuration
st.set_page_config(layout="wide")

st.title("Fruit Quality Detector")

# Sidebar: Display sample fruits
st.sidebar.header("Sample Fruits")
st.sidebar.write("Drag and drop images from below for classification.")

# Use columns in the sidebar to align images with spacing
st.sidebar.write("### Fresh Fruits")
cols = st.sidebar.columns(2)  # Create 2 columns for images in a row

# Fresh fruits
fresh_images = ["images/banana_good.JPG", "images/apple_good.jpg", "images/orangee_good.JPG", "images/pomogranate_good.jpg"]
fresh_captions = ["Good Banana", "Good Apple", "Good Orange", "Good Pomegranate"]

for idx, img_path in enumerate(fresh_images):
    with cols[idx % 2]:  # Cycle through columns
        st.image(img_path, caption=fresh_captions[idx], use_container_width=True)

st.sidebar.write("### Spoiled Fruits")
cols = st.sidebar.columns(2)  # Create 2 columns for images in a row

# Spoiled fruits
spoiled_images = ["images/babana_bad.JPG", "images/apple_bad (2).jpg", "images/orange_bad.jpg", "images/pomogranate_bad.jpg"]
spoiled_captions = ["Spoiled Banana", "Spoiled Apple", "Spoiled Orange", "Spoiled Pomegranate"]

for idx, img_path in enumerate(spoiled_images):
    with cols[idx % 2]:  # Cycle through columns
        st.image(img_path, caption=spoiled_captions[idx], use_container_width=True)

# Image Upload
input_img = st.file_uploader("Upload or Drag & Drop an image of a fruit", type=["jpg", "png", "jpeg"])

if input_img is not None:
    if st.button("Classify"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Your Uploaded Image")
            st.image(input_img, use_container_width=False, width=200)  # Smaller image

        with col2:
            st.info("Classification Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_fruit(image_file)

            if label.startswith("0 Good") or label.startswith("1 Good") or label.startswith("2 Good") or label.startswith("3 Good"):
                fruit_name = label.split(" ")[1]
                st.success(f"The fruit is: {fruit_name}")
                st.info("Condition: Good (Not Spoiled)")
            elif label.startswith("4") or label.startswith("5") or label.startswith("6") or label.startswith("7"):
                fruit_name = label.split(" ")[1]
                st.warning(f"The fruit is: {fruit_name}")
                st.error("Condition: Spoiled")
            else:
                st.error("The image could not be classified into any relevant category.")
