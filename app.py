from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Initialize session state for navigation
# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Test"

def navigate_to(page):
    st.session_state.page = page

# Function to display CNN info
def show_cnn_page():
    st.header("1. CNN (Convolutional Neural Networks)")
    st.subheader("(a) Architecture and Models")
    st.write("""
    A Convolutional Neural Network (CNN) is a deep learning algorithm specifically designed for processing structured grid data, like images.
    
    **Key Components:**
    - **Input Layer**: Holds raw pixel values.
    - **Convolutional Layer**: Uses filters to extract features like edges and textures.
    - **Activation Layer (ReLU)**: Applies non-linear functions to help the model learn complex patterns.
    - **Pooling Layer**: Reduces the spatial dimensions (e.g., Max Pooling) to decrease computation and prevent overfitting.
    - **Fully Connected (FC) Layer**: Connects every neuron in one layer to every neuron in another, usually for classification tasks.
    
    **Famous Models:**
    - **LeNet-5**: Digit recognition.
    - **AlexNet**: ImageNet 2012 winner.
    - **VGGNet**: 3x3 filter simplicity.
    - **Inception**: Multi-filter size modules.
    """)
    
    st.subheader("(b) How it Works")
    st.write("""
    Learns spatial hierarchies of features by scanning filters over images (convolution). 
    Early layers detect simple patterns like edges, while deeper layers recognize complex objects.
    """)
    
    st.subheader("(c) All the Steps Involved")
    st.markdown("""
    1. **Convolution**: Creating Feature Maps.
    2. **ReLU**: Non-linearity.
    3. **Pooling**: Downsampling.
    4. **Flattening**: 2D to 1D vector.
    5. **Full Connection**: Final classification.
    """)

# Function to display ResNet info
def show_resnet_page():
    st.header("2. ResNet (Residual Networks)")
    st.subheader("(a) Architecture and Diagrams")
    st.write("""
    ResNet solves the **Vanishing Gradient Problem** using **Skip Connections**.
    Instead of learning $H(x)$, it learns $F(x) = H(x) - x$. The output is $F(x) + x$.
    """)
    
    st.markdown("""
    ```text
    Input (x) ----+------> [ Weight Layer ] ----> [ ReLU ] ----> [ Weight Layer ] ----+---> Output (H(x))
                  |                                                                  ^
                  |                                                                  |
                  +------------------------- Shortcut / Identity --------------------+
    ```
    """)
    
    st.subheader("(b) Images and Related Information")
    st.write("""
    **ResNet-50** is widely used for transfer learning. It enables training of extremely deep networks without accuracy degradation.
    """)
    st.image("project.png", caption="ResNet Architecture skip connections.")

# Load models once using cache to improve performance
@st.cache_resource
def get_custom_model():
    return load_model("keras_model.h5", compile=False)

@st.cache_resource
def get_resnet_model():
    return ResNet50(weights='imagenet')

# Function to classify the fruit using custom model
def classify_fruit(img):
    np.set_printoptions(suppress=True)  # Disable scientific notation

    # Load the model
    model = get_custom_model()

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

# Function to classify using ResNet50
def classify_fruit_resnet(img):
    model = get_resnet_model()
    
    # Convert image to RGB and resize to 224x224 for ResNet
    image = img.convert("RGB")
    image = image.resize((224, 224))
    
    # Preprocess image for ResNet50
    x = np.asarray(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Predict
    preds = model.predict(x)
    # Return decodes top prediction (class_name, description, score)
    decoded = decode_predictions(preds, top=1)[0][0]
    return decoded[1], decoded[2]  # Return description and confidence

st.set_page_config(layout="wide", page_title="Fruit Quality & AI Info")

# Navigation logic
if st.session_state.page == "Test":
    # Sidebar for Test Page
    
    st.title("Fruit Quality Detector")
    
    # Rest of the Test page rendering happens later in the file
else:
    # Sidebar for Learn Section (reserved for other things as per user hint, but no nav buttons)
    st.sidebar.title("Deep Learning Info")
    st.sidebar.divider()

    # Render Learn Page (CNN & ResNet combined)
    st.title("Deep Learning: CNN & ResNet")
    
    tab1, tab2 = st.tabs(["CNN Architecture", "ResNet Architecture"])
    with tab1:
        show_cnn_page()
    with tab2:
        show_resnet_page()
    
    st.divider()
    # Bottom right button to go back to Test Section
    col1, col2, col3 = st.columns([4, 4, 2])
    with col3:
        if st.button("🍎 Go to Test Section →"):
            navigate_to("Test")
            st.rerun()

    st.stop() # Prevents rendering the Test page content

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
        st.image(img_path, caption=fresh_captions[idx], use_column_width=True)

st.sidebar.write("### Spoiled Fruits")
cols = st.sidebar.columns(2)  # Create 2 columns for images in a row

# Spoiled fruits
spoiled_images = ["images/babana_bad.JPG", "images/apple_bad (2).jpg", "images/orange_bad.jpg", "images/pomogranate_bad.jpg"]
spoiled_captions = ["Spoiled Banana", "Spoiled Apple", "Spoiled Orange", "Spoiled Pomegranate"]

for idx, img_path in enumerate(spoiled_images):
    with cols[idx % 2]:  # Cycle through columns
        st.image(img_path, caption=spoiled_captions[idx], use_column_width=True)

# Image Upload
input_img = st.file_uploader("Upload or Drag & Drop an image of a fruit", type=["jpg", "png", "jpeg", "webp"])

if input_img is not None:
    if st.button("Classify"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Your Uploaded Image")
            st.image(input_img, use_column_width=False, width=200)  # Smaller image

        with col2:
            st.info("Classification Result")
            try:
                image_file = Image.open(input_img)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.info("Ensure the file is a valid JPG, PNG, or WEBP image.")
                st.stop()
            
            label, confidence_score = classify_fruit(image_file)

            if label.startswith("0 Good") or label.startswith("1 Good") or label.startswith("2 Good") or label.startswith("3 Good"):
                # Format is "index Good FruitName"
                fruit_name = label.split(" ")[2]
                st.success(f"**Custom Model Prediction**: {fruit_name}")
                st.info(f"Condition: Good (Not Spoiled)")
            elif label.startswith("4") or label.startswith("5") or label.startswith("6") or label.startswith("7"):
                # Format is "index FruitName Bad"
                fruit_name = label.split(" ")[1]
                st.warning(f"**Custom Model Prediction**: {fruit_name}")
                st.error("Condition: Spoiled")
            else:
                st.error("The custom model could not classify this image.")
            
            st.code(f"Confidence (Custom): {confidence_score:.2%}")

            st.divider()

            # ResNet50 Result
            st.info("ResNet50 Prediction (General Purpose)")
            resnet_label, resnet_score = classify_fruit_resnet(image_file)
            st.write(f"**Object Detected**: {resnet_label.replace('_', ' ').title()}")
            st.code(f"Confidence (ResNet50): {resnet_score:.2%}")
            
            st.caption("Note: ResNet50 is a general-purpose model trained on ImageNet, while the Custom Model is specialized for fruit quality.")

st.divider()
# Bottom right button for navigation to Learn Section
col1, col2, col3 = st.columns([4, 4, 2])
with col3:
    if st.button("🧠 Learn CNN ResNet →"):
        navigate_to("Learn")
        st.rerun()
