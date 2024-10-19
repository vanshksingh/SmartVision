import cv2
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from apple_ocr.ocr import OCR
import ollama  # Ensure you have the correct import for Ollama
import logging
import os
from datetime import datetime

# --------------------- Configuration ---------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("real_time_detection.log"),
        logging.StreamHandler()
    ]
)

# Configuration dictionary
CONFIG = {
    "YOLO_MODEL_PATH": "yolo11n.pt",
    "RESNET_MODEL_PATH": "banana_tomato_classifier.pth",
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
    "PROCESS_OCR_EVERY_N_FRAMES": 5  # Process OCR every N frames to optimize performance
}

# --------------------- Model Loading ---------------------

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

# --------------------- Utility Functions ---------------------

# Function to predict class and freshness index
def predict_image_with_freshness_index(image, model, transform, class_names):
    try:
        # Ensure transform is callable
        if not callable(transform):
            raise TypeError(f"Transform is not callable. Received type: {type(transform)}")

        # Transform the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Disable gradient calculations for inference
        with torch.no_grad():
            outputs = model(image)  # Get model outputs
            probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get the highest confidence

        # Calculate freshness index (assuming class 0 is 'banana ripe')
        freshness_index = probabilities[0, 0].item() * 10  # Scale to range (0 to 10)

        predicted_class = class_names[predicted.item()]

        return predicted_class, freshness_index
    except Exception as e:
        logging.error(f"Error in predict_image_with_freshness_index: {e}")
        return "Unknown", 0.0

# Function to perform OCR and get product details using LLMs (Ollama)
def detect_product_info(image_region):
    try:
        # Perform OCR on the image region
        ocr_image = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
        ocr_instance = OCR(image=ocr_image)
        dataframe = ocr_instance.recognize()

        # Check if the OCR results contain a 'Content' column
        if 'Content' in dataframe.columns and not dataframe.empty:
            detected_text = " ".join(dataframe['Content'].tolist())
        else:
            detected_text = ""

        # If no text was detected, return early
        if detected_text.strip() == "":
            return "No product info found", False

        # Query the LLM to extract product info (name, expiry, etc.)
        query = f"Extract product name and expiry information from this text: {detected_text}"

        # Using Ollama (ensure you have the correct API usage)
        ollama_response = ollama.run("qwen2.5:0.5b-instruct", query)

        # Process the LLM response to determine if expired
        product_info = ollama_response.strip()

        # Check if the product is expired based on the extracted information
        if "expired" in product_info.lower():
            return product_info, True  # Product is expired
        return product_info, False  # Product is not expired

    except Exception as e:
        # Handle any exceptions that occur during OCR or LLM interaction
        logging.error(f"Error in detect_product_info: {e}")
        return "Error detecting product info", False

# --------------------- Real-Time Detection Function ---------------------

def real_time_detection(yolo_model, resnet_model, transform, config):
    cap = cv2.VideoCapture(config["CAMERA_INDEX"])  # Open the camera

    frame_count = 0  # To control OCR processing frequency

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame from the camera

        if not ret:
            logging.warning("Failed to grab frame")
            break

        # Run YOLO inference on the captured frame
        results = yolo_model(frame)

        for result in results:
            boxes = result.boxes  # Detected bounding boxes
            for box in boxes:
                try:
                    # Get the class index of the detected object
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]  # Get the name of the detected class

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers for OpenCV

                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    # Default label and color
                    label = ""
                    color = config["LABEL_COLOR_DEFAULT"]

                    # Priority Handling: Check if the detected object is a fruit
                    if class_name.lower() in [fruit.lower() for fruit in config["FRUITS"]]:
                        # Crop the detected fruit from the frame
                        cropped_object = frame[y1:y2, x1:x2]

                        # Convert the cropped image to PIL format
                        cropped_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

                        # Use ResNet model to classify and calculate freshness index
                        predicted_class, freshness_index = predict_image_with_freshness_index(
                            cropped_image, resnet_model, transform, config["CLASS_NAMES"]
                        )

                        # Change the bounding box color to red if the fruit is classified as rotten
                        if "rotten" in predicted_class.lower():
                            color = config["LABEL_COLOR_ALERT"]

                        # Set the label with classification and freshness index
                        label = f"{predicted_class}: Freshness {freshness_index:.2f}/10"

                    else:
                        # For generic items, process OCR every N frames to optimize performance
                        if frame_count % config["PROCESS_OCR_EVERY_N_FRAMES"] == 0:
                            # Crop the generic item from the frame
                            cropped_object = frame[y1:y2, x1:x2]

                            # Perform OCR and get product info
                            product_info, is_expired = detect_product_info(cropped_object)

                            # Set color to red if the product is expired
                            if is_expired:
                                color = config["LABEL_COLOR_ALERT"]

                            # Set the label with "generic" and YOLO class name
                            label = f"generic ({class_name})"
                        else:
                            # Skip OCR processing for this frame to save resources
                            label = f"generic ({class_name})"

                    # Draw the bounding box around the object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add label on the video frame
                    if len(label) > config["MAX_LABEL_LENGTH"]:
                        label = label[:config["MAX_LABEL_LENGTH"] - 3] + "..."
                    cv2.putText(frame, label, (x1, y1 - 10), config["FONT"],
                                config["LABEL_FONT_SCALE"], (255, 255, 255), config["LABEL_THICKNESS"], cv2.LINE_AA)

                except Exception as e:
                    logging.error(f"Error processing box: {e}")
                    continue

        # Increment frame count
        frame_count += 1

        # Display the result in a window
        cv2.imshow("YOLO + ResNet + OCR Real-Time Detection", frame)

        # Press 'q' to quit the real-time detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Quitting real-time detection.")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# --------------------- Main Execution ---------------------

if __name__ == "__main__":
    try:
        # Load models
        yolo_model = load_yolo_model(CONFIG["YOLO_MODEL_PATH"])
        resnet_model = load_resnet_model(CONFIG["RESNET_MODEL_PATH"], CONFIG["CLASS_NAMES"])

        # Define the transform for the ResNet model
        test_transform = transforms.Compose([
            transforms.Resize(CONFIG["OCR_IMAGE_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Start real-time object detection and classification
        logging.info("Starting real-time detection...")
        real_time_detection(yolo_model, resnet_model, test_transform, CONFIG)

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
