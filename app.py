from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st
import os

# Initialize session state for navigation
# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Test"

def navigate_to(page):
    st.session_state.page = page

if "learn_page" not in st.session_state:
    st.session_state.learn_page = "About Project"

def navigate_learn(sub_page):
    st.session_state.learn_page = sub_page

# Session state for Sidebar Gallery
if "gallery_visible" not in st.session_state:
    st.session_state.gallery_visible = False
if "gallery_count" not in st.session_state:
    st.session_state.gallery_count = 10
if "run_batch" not in st.session_state:
    st.session_state.run_batch = False

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

# Function to display ResNet-50 Specific Info
def show_resnet50_detailed():
    st.header("ResNet-50: A Closer Look")
    st.write("""
    ResNet-50 is a variant of the ResNet (Residual Network) model that has 50 layers deep. 
    The '50' refers to the number of weighted layers (48 convolutional layers, 1 connected layer, and 1 pool layer).
    
    **Key Technical Specifications:**
    - **Convolutional Layers**: 48 layers for feature extraction.
    - **Fully Connected Layer**: 1 layer for classification.
    - **Average Pooling Layer**: 1 layer for dimensionality reduction.
    - **Number of Parameters**: Roughly 25.6 million.
    - **Input Size**: Standard size is 224x224x3 (RGB images).
    
    **Why use ResNet-50?**
    It balances depth and performance. While deeper models exist (ResNet-101, ResNet-152), 
    ResNet-50 is often the benchmark for transfer learning because it is accurate enough for most tasks while being computationally efficient.
    """)
    st.info("In this project, ResNet-50 is used as a baseline comparator to validate the performance of our custom-trained CNN model.")

# Function to display project info and process
def show_about_project():
    st.header("What my project is about")
    st.write("""
    This project is a **Fruit Classification and Quality Detection System**. 
    It aims to automate the process of identifying various fruits and determining whether they are fresh or spoiled. 
    By leveraging deep learning models, we can provide consistent and rapid assessments of fruit quality, 
    which is essential for the food industry and supply chain management.
    """)
    
    st.header("Development Process")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("(a) Training Process")
        st.markdown("""
        1. **Data Collection**: Gathering thousands of images of fresh and spoiled fruits.
        2. **Preprocessing**: Resizing images to 224x224, normalizing pixel values, and data augmentation.
        3. **Model Selection**: choosing a CNN architecture suitable for image classification.
        4. **Training**: Using backpropagation to minimize classification error.
        5. **Validation**: Testing on unseen data to ensure the model generalizes well.
        """)
    with col2:
        st.subheader("(b) How we test it")
        st.markdown("""
        **Step-by-Step Flow:**
        1. **Upload**: User uploads a fruit image or drags it from the samples.
        2. **CNN Model**: Our specialized custom model classifies the fruit species and quality (Good/Bad).
        3. **ResNet-50**: A pre-trained ResNet-50 model provides a general-purpose prediction for verification.
        4. **Output**: The system displays the detected object, condition, and confidence scores for both models.
        """)

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
    st.sidebar.header("Sample Fruits")
    st.sidebar.write("Drag and drop images for classification.")
    
    st.sidebar.write("### Fresh Fruits")
    f_cols = st.sidebar.columns(2)
    fresh_images = ["images/banana_good.JPG", "images/apple_good.jpg", "images/orangee_good.JPG", "images/pomogranate_good.jpg"]
    fresh_captions = ["Good Banana", "Good Apple", "Good Orange", "Good Pomegranate"]
    for idx, img_path in enumerate(fresh_images):
        with f_cols[idx % 2]:
            st.image(img_path, caption=fresh_captions[idx], use_column_width=True)

    st.sidebar.write("### Spoiled Fruits")
    s_cols = st.sidebar.columns(2)
    spoiled_images = ["images/babana_bad.JPG", "images/apple_bad (2).jpg", "images/orange_bad.jpg", "images/pomogranate_bad.jpg"]
    spoiled_captions = ["Spoiled Banana", "Spoiled Apple", "Spoiled Orange", "Spoiled Pomegranate"]
    for idx, img_path in enumerate(spoiled_images):
        with s_cols[idx % 2]:
            st.image(img_path, caption=spoiled_captions[idx], use_column_width=True)

    st.sidebar.divider()
    
    # Local based testing section in sidebar
    st.sidebar.header("📁 Local-based Testing")
    st.sidebar.write("Browse and test images from `./random` folder.")

    # Show/Hide Gallery
    if not st.session_state.gallery_visible:
        if st.sidebar.button("👁️ View Random Images"):
            st.session_state.gallery_visible = True
            st.rerun()
    else:
        st.sidebar.subheader("Gallery Preview")
        random_dir = "random"
        if os.path.exists(random_dir):
            all_files = [f for f in os.listdir(random_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            all_files.sort()
            
            show_count = st.session_state.gallery_count
            current_files = all_files[:show_count]
            
            g_cols = st.sidebar.columns(2)
            for i, f in enumerate(current_files):
                with g_cols[i % 2]:
                    st.image(os.path.join(random_dir, f), use_column_width=True)
            
            # Gallery Controls
            c1, c2 = st.sidebar.columns(2)
            if show_count < len(all_files):
                if c1.button("➕ More"):
                    st.session_state.gallery_count += 10
                    st.rerun()
            
            if c2.button("❌ Close Preview"):
                st.session_state.gallery_visible = False
                st.session_state.gallery_count = 10
                st.rerun()
        else:
            st.sidebar.error("random/ not found.")

    st.sidebar.divider()
    # Run Batch button in sidebar
    if st.sidebar.button("🚀 Run Batch Classifier"):
        st.session_state.run_batch = True
        st.rerun()

    st.title("Fruit Quality Detector")
    
    # Rest of the Test page rendering happens later in the file
else:
    # Sidebar for Learn Section
    st.sidebar.title("📚 Learn Section")
    
    if st.sidebar.button("📄 About Project"):
        navigate_learn("About Project")
    if st.sidebar.button("🔍 CNN ResNet Overview"):
        navigate_learn("Overview")
    if st.sidebar.button("🚀 ResNet-50 Deep Dive"):
        navigate_learn("Deep Dive")
    if st.sidebar.button("🏗️ Architecture"):
        navigate_learn("Architecture")
        
    st.sidebar.divider()
    
    # Back to Classification button in sidebar
    if st.sidebar.button("🍎 Back to Classification"):
        navigate_to("Test")
        st.rerun()

    # Render Learn Page Content based on selection
    if st.session_state.learn_page == "About Project":
        show_about_project()
    elif st.session_state.learn_page == "Overview":
        st.title("Deep Learning Overview")
        show_cnn_page()
        st.divider()
        show_resnet_page()
    elif st.session_state.learn_page == "Deep Dive":
        show_resnet50_detailed()
    elif st.session_state.learn_page == "Architecture":
        st.title("Project Architecture")
        st.subheader("Visualizing the Process")
        st.info("The system follows a sequential pipeline from input to result.")
        st.markdown("""
        1. **Input**: Image Data (Matrice of pixels)
        2. **Convolution**: Feature Extraction (Edges, Shapes)
        3. **Classification**: Dense Layers matching features to labels
        4. **Output**: Probability Score (e.g., 98% Good Pomegranate)
        """)
    
    st.divider()
    # Bottom right button to go back to Test Section (preserving linear flow)
    col1, col2, col3 = st.columns([4, 3, 3])
    with col3:
        if st.button("🍎 Go to Test Section →"):
            navigate_to("Test")
            st.rerun()

    st.stop() # Prevents rendering the Test page content


# Bulk Image Upload
input_imgs = st.file_uploader("Upload or Drag & Drop multiple fruit images", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True)

if input_imgs:
    if st.button("Classify Images"):
        st.subheader("Classification Results Table")
        
        # Table Header
        header_cols = st.columns([1, 2, 2])
        header_cols[0].markdown("**Input Image**")
        header_cols[1].markdown("**Custom CNN Model Result**")
        header_cols[2].markdown("**ResNet-50 Result**")
        st.divider()

        for input_img in input_imgs:
            try:
                image_file = Image.open(input_img)
            except Exception as e:
                st.error(f"Error loading image {input_img.name}: {e}")
                continue
            
            # Row for each image
            row_cols = st.columns([1, 2, 2])
            
            # Column 1: Image
            with row_cols[0]:
                st.image(input_img, use_column_width=True)
            
            # Run Classifications
            label, confidence_score = classify_fruit(image_file)
            resnet_label, resnet_score = classify_fruit_resnet(image_file)
            
            # Column 2: CNN Result
            with row_cols[1]:
                if label.startswith("0 Good") or label.startswith("1 Good") or label.startswith("2 Good") or label.startswith("3 Good"):
                    fruit_name = label.split(" ")[2]
                    st.success(f"**Species**: {fruit_name}")
                    st.info(f"**Condition**: Fresh")
                elif label.startswith("4") or label.startswith("5") or label.startswith("6") or label.startswith("7"):
                    fruit_name = label.split(" ")[1]
                    st.warning(f"**Species**: {fruit_name}")
                    st.error("**Condition**: Spoiled")
                else:
                    st.error("Unclassified")
                st.write(f"Confidence: {confidence_score:.1%}")

            # Column 3: ResNet Result
            with row_cols[2]:
                st.write(f"**Object**: {resnet_label.replace('_', ' ').title()}")
                st.write(f"Confidence: {resnet_score:.1%}")
                
            st.divider()
        
        st.divider()

# Logic for Batch Classification (Triggered from Sidebar)
if st.session_state.run_batch:
    # Reset the flag after starting
    st.session_state.run_batch = False
    
    st.subheader("🔥 Random Folder Batch Classification Results")
    random_dir = "random"
    if os.path.exists(random_dir):
        files = [f for f in os.listdir(random_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if files:
            st.info(f"Processing {len(files)} images from `./random`...")
            
            # Use columns for table
            header_cols = st.columns([1, 2, 2])
            header_cols[0].markdown("**Input Image**")
            header_cols[1].markdown("**Custom CNN Model Result**")
            header_cols[2].markdown("**ResNet-50 Result**")
            st.divider()

            for filename in files:
                file_path = os.path.join(random_dir, filename)
                try:
                    image_file = Image.open(file_path)
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
                    continue
                
                row_cols = st.columns([1, 2, 2])
                with row_cols[0]:
                    st.image(file_path, use_column_width=True)
                
                label, confidence_score = classify_fruit(image_file)
                resnet_label, resnet_score = classify_fruit_resnet(image_file)
                
                with row_cols[1]:
                    if label.startswith("0 Good") or label.startswith("1 Good") or label.startswith("2 Good") or label.startswith("3 Good"):
                        fruit_name = label.split(" ")[2]
                        st.success(f"**Species**: {fruit_name}")
                        st.info("**Condition**: Fresh")
                    elif label.startswith("4") or label.startswith("5") or label.startswith("6") or label.startswith("7"):
                        fruit_name = label.split(" ")[1]
                        st.warning(f"**Species**: {fruit_name}")
                        st.error("**Condition**: Spoiled")
                    else:
                        st.error("Unclassified")
                    st.write(f"Confidence: {confidence_score:.1%}")

                with row_cols[2]:
                    st.write(f"**Object**: {resnet_label.replace('_', ' ').title()}")
                    st.write(f"Confidence: {resnet_score:.1%}")
                st.divider()
    else:
        st.error("`random/` folder not found.")

st.caption("Note: Custom Model is specialized for fruit quality; ResNet-50 is for general-purpose validation.")
st.divider()



# Bottom right button for navigation to Learn Section
col1, col2, col3 = st.columns([4, 3, 3])
with col3:
    if st.button("🧠 Learn about the Project & AI →"):
        navigate_to("Learn")
        st.rerun()
