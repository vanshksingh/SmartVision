# config.py

import cv2

# --------------------- Configuration ---------------------

CONFIG = {
    "YOLO_MODEL_PATH": "/Users/vanshkumarsingh/Desktop/Flipkart/yolo11n.pt",
    "RESNET_MODEL_PATH": "/Users/vanshkumarsingh/Desktop/Flipkart/banana_tomato_classifier.pth",
    "FRUITS": ['banana'],  # Add more fruits if needed
    "CLASS_NAMES": ['banana ripe', 'banana rotten'],  # ResNet class names
    "OCR_IMAGE_SIZE": (224, 224),  # Resize size for ResNet
    "CAMERA_INDEX": 0,  # Default camera
    "FONT": cv2.FONT_HERSHEY_SIMPLEX,
    "LABEL_COLOR_DEFAULT": (0, 255, 0),  # Green
    "LABEL_COLOR_ALERT": (0, 0, 255),    # Red
    "LABEL_FONT_SCALE": 0.6,
    "LABEL_THICKNESS": 2,
     "MAX_LABEL_LENGTH": 30,
    "PROCESS_OCR_EVERY_N_FRAMES": 5,  # Process OCR every N frames to optimize performance
    "TRACKER_EXPIRATION_FRAMES": 30   # Number of frames to track an object before expiring
}