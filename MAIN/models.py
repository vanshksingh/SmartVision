# models.py

import os
import logging
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import torch.nn as nn

def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        logging.error(f"YOLO model file not found at {model_path}")
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
    logging.info("Loading YOLO model...")
    yolo_model = YOLO(model_path)
    logging.info("YOLO model loaded successfully.")
    return yolo_model

def load_resnet_model(model_path, class_names):
    if not os.path.exists(model_path):
        logging.error(f"ResNet model file not found at {model_path}")
        raise FileNotFoundError(f"ResNet model file not found at {model_path}")
    logging.info("Loading ResNet model...")
    resnet_model = models.resnet18(weights=None)  # Start from scratch
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjust for number of classes
    resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load on CPU
    resnet_model.eval()
    logging.info("ResNet model loaded successfully.")
    return resnet_model

def get_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform
