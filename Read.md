# **AI Vision Quality Checker: Freshness & Product Quality Detection**

Welcome to the **AI Vision Quality Checker** project! This repository is designed to automate product quality control using state-of-the-art computer vision and machine learning techniques. The system leverages YOLO for real-time object detection, ResNet for product classification (such as freshness detection), OCR for extracting textual data like brand names and expiration dates, and advanced language models (LLMs) for analyzing product details. It’s an end-to-end solution for quality control, particularly for e-commerce applications, where accuracy, efficiency, and scalability are paramount.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Hardware Specifications](#hardware-specifications)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [How It Works](#how-it-works)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Overview**
This project aims to create a real-time quality control system capable of detecting:
- **Product Freshness**: Identifying fresh vs. stale products (e.g., fruits and vegetables).
- **Textual Information**: Extracting details such as brand name, expiration date, and package size from product labels.
- **Product Classification**: Classifying various types of packaged goods and fresh produce.
- **Real-time Tracking & Updates**: Using object tracking to maintain consistent labeling and periodically updating details in case the OCR missed something.

---

## **Features**
- **Real-Time Object Detection**: Uses YOLO to identify objects and track them across frames.
- **Product Freshness Detection**: Uses ResNet to classify fruits and vegetables as fresh or stale, with a freshness index.
- **OCR for Text Extraction**: Extracts important product details such as brand names, expiry dates, and packaging size using Apple’s OCR.
- **Language Model Integration**: Utilizes Ollama's large language model to analyze and validate product details from OCR.
- **Continuous Tracking**: Tracks objects over multiple frames using the SORT algorithm to ensure accuracy over time.

---

## **Technology Stack**

### **Core Technologies**:
- **Python 3.x**: Programming language.
- **YOLO (Ultralytics)**: Used for real-time object detection.
- **PyTorch & ResNet**: For training and running deep learning models to classify products and detect freshness.
- **Apple OCR**: Extracts textual information from product labels such as brand names and expiration dates.
- **Ollama Language Models**: Used to interpret and extract meaningful details from the OCR output.
- **SORT Algorithm**: For tracking objects across frames to ensure consistent product labeling and updates.

### **Key Python Libraries**:
```python
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from apple_ocr.ocr import OCR
import ollama
import numpy as np
from sort import Sort  # For object tracking
```

---

## **Hardware Specifications**
This project has been developed and tested on the following hardware:

- **M2 MacBook Pro (8GB RAM)**: Used for local development, inference, and real-time image processing.
- **1080p Cameras**: High-resolution cameras to capture detailed product images.
- **LED Lighting Setup**: To ensure consistent illumination for improved OCR and object detection accuracy.
  
Note: This setup is scalable to more powerful hardware, such as NVIDIA GPUs, for high-throughput operations.

---

## **Dataset**
We use a custom dataset to train the freshness detection model, which consists of images of fresh and stale fruits and vegetables.

### **Dataset: Fresh and Stale Classification**
We are using the **Fresh and Stale Classification Dataset** available on Kaggle, which can be downloaded using the link below:

[**Kaggle Dataset: Fresh and Stale Classification**](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)

This dataset contains images of fresh and stale fruits, along with labels indicating their quality. It is used to train the ResNet model for classifying products into fresh or stale categories.

### **Dataset Details**:
- **Fresh Images**: 850+ images of fresh produce.
- **Stale Images**: 600+ images of stale produce.
- **Classes**: Two classes: **Fresh** and **Stale**.

---

## **Installation**

### **1. Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/ai-vision-quality-checker.git
cd ai-vision-quality-checker
```

### **2. Install Required Dependencies**
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes all necessary libraries such as:
- OpenCV
- PyTorch
- Ultralytics YOLO
- PIL
- SORT (Object Tracking)
- Apple OCR
- Ollama

### **3. Download Dataset**
Download the **Fresh and Stale Classification** dataset from [Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification), extract it, and place it in the `data/` folder within the project directory.

---

## **How It Works**

### **1. Image Acquisition**:
Cameras capture high-resolution images of products in real-time from conveyor belts or inspection stations.

### **2. Object Detection (YOLO)**:
- YOLO detects the products within each frame and identifies bounding boxes around them.
- It classifies the object type (e.g., fruit, vegetable, packaged good).

### **3. Freshness Detection (ResNet)**:
- The ResNet model processes images of fruits and vegetables to classify them as **fresh** or **stale**.
- A **freshness index** (scaled from 0 to 10) is calculated for each detected product.

### **4. OCR for Text Extraction**:
- Apple OCR extracts textual information like **brand names**, **expiry dates**, and **package size** from product labels.
- This information is then processed to determine if the product is expired.

### **5. Language Model Processing**:
- The extracted text is passed to **Ollama**, which processes the text and infers meaningful product details (e.g., verifying expiration dates, brand recognition).

### **6. Object Tracking (SORT Algorithm)**:
- **SORT** tracks detected objects across multiple frames, ensuring the labels and product information are updated accurately over time.
- If OCR misses something in earlier frames, the system updates it with new information in subsequent frames.

### **7. Continuous Updates**:
- Periodically checks for missing details and updates the labels as new information is gathered from the frames.

---

## **Usage**

### **1. Real-Time Detection**
Run the following command to start the real-time object detection and quality assessment:
```bash
python detection.py
```

This script will:
- Open the camera feed.
- Use YOLO for real-time object detection.
- Classify freshness using the ResNet model.
- Extract text using OCR and process it with Ollama.
- Display the live feed with bounding boxes, product labels, freshness indices, and OCR-extracted text.

### **2. Testing with Images**
To test the system with sample images instead of the camera, run:
```bash
python test_with_image.py --image-path /path/to/your/image.jpg
```
This will simulate the quality control process on a static image.

---

## **Contributing**
We welcome contributions to improve the project. If you’d like to contribute, please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

Feel free to open issues for feature requests, bug reports, or general feedback.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---


We hope you find this project useful, and we look forward to your contributions!
