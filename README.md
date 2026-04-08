# 🍎 Fruit Quality Detector: Technical Architecture & Dual-Engine AI

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

---

## 📖 Introduction
The **Fruit Quality Detector** is a high-performance computer vision application. It solves the critical task of distinguishing between **Good** and **Spoiled** produce by analyzing visual features such as skin texture, color uniformity, and surface defects.

---

## 🧠 Core Concept: Convolutional Neural Networks (CNN)

### **What is a CNN?**
A **Convolutional Neural Network** is a deep learning architecture inspired by the human visual cortex. Unlike standard neural networks that see images as a flat list of pixels, a CNN understands **spatial hierarchies** (edges → shapes → objects).

#### **The 4 Key Layers in our CNN:**
1.  **Convolutional Layer**: Uses "filters" to scan the image. It acts like a magnifying glass looking for specific patterns (like a brown spot on an apple).
2.  **Activation (ReLU)**: Adds non-linearity. It decides which features are "important" enough to pass to the next layer.
3.  **Pooling (Max Pooling)**: Reduces the image size while keeping the most important information. This makes the model faster and more robust to image rotation.
4.  **Dense (Fully Connected)**: The final "brain" that takes all the extracted features and makes the final decision: *Is this a good orange or a spoiled one?*

---

## 🏛️ Custom Model: MobileNetV2 Backbone

### **What is a "Backbone"?**
In deep learning, a **Backbone** is a pre-trained model that acts as a "Feature Extractor." We use **MobileNetV2** as our backbone. It has already "seen" millions of images (ImageNet) and knows how to recognize shapes, colors, and textures.

*   **Why MobileNetV2?**: It is designed for speed. It uses *Depthwise Separable Convolutions* to provide high accuracy while using very little memory.
*   **The Custom Head**: We removed the original classification layer of MobileNetV2 and added our own **Custom Layers** (Dense, Dropout, Softmax) to specifically detect fruit quality.

---

## 🧬 ResNet-50: Residual Architecture

### **Advanced Architecture Breakdown**
ResNet (Residual Network) is famous for its **Skip Connections**. 

Standard networks try to learn the full mapping $H(x)$. ResNet instead learns the "Residual" $F(x) = H(x) - x$. This allows the network to effectively "bypass" layers if they aren't helping, which prevents the **Vanishing Gradient Problem** (where the model stops learning because it's too deep).

#### **ResNet-50 Data Flow Diagram**
```mermaid
graph TD
    subgraph "Phase 1: Stem"
    In["Input (224x224x3)"] --> C1["7x7 Conv (64 Filters)"]
    C1 --> BN1["Batch Norm + ReLU"]
    BN1 --> MP1["3x3 Max Pool"]
    end

    subgraph "Phase 2: Residual Stages"
    MP1 --> S1["Stage 1: 3x Bottleneck Blocks"]
    S1 --> S2["Stage 2: 4x Bottleneck Blocks"]
    S2 --> S3["Stage 3: 6x Bottleneck Blocks"]
    S3 --> S4["Stage 4: 3x Bottleneck Blocks"]
    end

    subgraph "Phase 3: Classification Head"
    S4 --> GAP["Global Average Pooling"]
    GAP --> FC["Fully Connected (1000 units)"]
    FC --> SM["Softmax (Probabilities)"]
    end

    style In fill:#f9f9f9,stroke:#333
    style SM fill:#dcedc8,stroke:#33691e,stroke-width:2px
```

---

## 📊 Models Table: Side-by-Side

| Feature | Custom CNN (MobileNetV2) | ResNet-50 |
| :--- | :--- | :--- |
| **Logic** | Specialized Fine-Tuning | Pre-trained Generalist |
| **Architecture** | Depthwise Separable Conv | Residual Skip Connections |
| **Primary Goal** | **Fruit Freshness** | **General Object Identity** |
| **Performance** | High Accuracy on this Dataset | Baseline Comparisons |

---

## 🚀 Setup & Execution

> [!NOTE]
> Ensure you have **TensorFlow 2.15.0** installed for maximum compatibility with the `.h5` model files.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit server
streamlit run app.py
```

---

<p align="center">
  <b>Developed for professional fruit quality assessment using Deep Learning.</b>
</p>
