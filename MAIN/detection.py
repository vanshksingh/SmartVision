# detection.py

import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import logging
import numpy as np

from models import load_yolo_model, load_resnet_model, get_transform
from ocr_llm import extract_product_details
from config import CONFIG

# Import the SORT tracker
from sort import Sort  # Ensure you have sort.py in your project directory

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list): [x1, y1, x2, y2] coordinates of the first box.
        boxB (list): [x1, y1, x2, y2] coordinates of the second box.

    Returns:
        float: IoU value.
    """
    # Calculate the intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    return iou

def predict_image_with_freshness_index(image, model, transform, class_names):
    """
    Predict the class and freshness index of an image using the ResNet model.

    Args:
        image (PIL.Image.Image): The image to predict.
        model (torch.nn.Module): The ResNet model.
        transform (callable): The image transformation function.
        class_names (list): List of class names.

    Returns:
        tuple: Predicted class and freshness index.
    """
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

def real_time_detection(yolo_model, resnet_model, transform, config):
    """
    Perform real-time object detection, classification, and OCR with tracking.

    Args:
        yolo_model: Loaded YOLO model.
        resnet_model: Loaded ResNet model.
        transform: Image transformation for ResNet.
        config: Configuration dictionary.
    """
    cap = cv2.VideoCapture(config["CAMERA_INDEX"])  # Open the camera

    frame_count = 0  # To control OCR processing frequency

    # Initialize SORT tracker
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # Dictionary to store labels associated with object IDs
    object_labels = {}

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame from the camera

        if not ret:
            logging.warning("Failed to grab frame")
            break

        # Run YOLO inference on the captured frame
        results = yolo_model(frame)

        detections = []

        for result in results:
            boxes = result.boxes  # Detected bounding boxes
            for box in boxes:
                # Get the class index of the detected object
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]  # Get the name of the detected class

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers for OpenCV

                # Confidence score
                confidence = box.conf[0].item()

                # Prepare detection in [x1, y1, x2, y2, confidence] format
                detections.append([x1, y1, x2, y2, confidence])

        # Convert detections to numpy array
        if len(detections) > 0:
            detections_np = np.array(detections)
        else:
            detections_np = np.empty((0, 5))

        # Update tracker with current frame detections
        tracked_objects = tracker.update(detections_np)

        # Prepare to associate class IDs with tracked objects
        tracked_objects_with_class = []

        for tracked_obj in tracked_objects:
            x1_t, y1_t, x2_t, y2_t, obj_id = map(int, tracked_obj[:5])
            # Initialize max IoU and class_id
            max_iou = 0
            matched_class_id = -1

            # Compare with detections to find the best match
            for det in detections:
                x1_d, y1_d, x2_d, y2_d, confidence = det
                iou = compute_iou([x1_t, y1_t, x2_t, y2_t], [x1_d, y1_d, x2_d, y2_d])
                if iou > max_iou:
                    max_iou = iou
                    matched_class_id = class_id  # Use the class_id from YOLO detection

            # Append the tracked object with the matched class ID
            tracked_objects_with_class.append([x1_t, y1_t, x2_t, y2_t, obj_id, matched_class_id])

        # Process tracked objects
        for obj in tracked_objects_with_class:
            x1, y1, x2, y2, obj_id, class_id = map(int, obj)
            class_name = yolo_model.names[class_id] if class_id >= 0 else "Unknown"

            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Default label and color
            label = ""
            color = config["LABEL_COLOR_DEFAULT"]

            # Check if object ID is already in the labels dictionary
            if obj_id not in object_labels:
                # Initialize label info
                object_labels[obj_id] = {
                    "label": "",
                    "timestamps": [],
                    "label_history": [],
                    "last_processed": -config["PROCESS_OCR_EVERY_N_FRAMES"]  # Ensure immediate processing
                }

            label_info = object_labels[obj_id]

            # Determine if it's time to reprocess OCR/LLM
            if frame_count - label_info["last_processed"] >= config["PROCESS_OCR_EVERY_N_FRAMES"]:
                # Update last processed frame
                label_info["last_processed"] = frame_count

                # For fruits, classify with ResNet
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
                    label_info["label"] = f"{predicted_class}: Freshness {freshness_index:.2f}/10"
                else:
                    # For generic items, perform OCR and LLM
                    # Crop the generic item from the frame
                    cropped_object = frame[y1:y2, x1:x2]

                    # Convert the cropped image to PIL format
                    cropped_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

                    # Extract product details using OCR and LLM
                    product_details = extract_product_details([cropped_image])

                    # Determine label based on extraction
                    if "error" not in product_details:
                        brand = product_details.get("brand_name", "Unknown")
                        expiry = product_details.get("expiry_date", "Unknown")
                        package = product_details.get("package_size", "Unknown")

                        label_info["label"] = f"{brand}, Package: {package} Expiry: {expiry}"

                        # Change color if expired
                        if expiry != "Unknown" and "expired" in expiry.lower():
                            color = config["LABEL_COLOR_ALERT"]
                    else:
                        label_info["label"] = f"generic ({class_name})"

            else:
                # Use existing label
                pass

            # Update timestamp for the object
            label_info["timestamps"].append(frame_count)
            if "label_history" not in label_info:
                label_info["label_history"] = []

            label_info["label_history"].append(label_info["label"])

            # Keep only the last N labels for smoothing
            N = 5  # Adjust as needed
            if len(label_info["label_history"]) > N:
                label_info["label_history"] = label_info["label_history"][-N:]

            # Use majority voting for label
            most_common_label = max(set(label_info["label_history"]), key=label_info["label_history"].count)
            label = most_common_label

            # Draw the bounding box around the object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label on the video frame
            if len(label) > config["MAX_LABEL_LENGTH"]:
                label = label[:config["MAX_LABEL_LENGTH"] - 3] + "..."
            cv2.putText(frame, f"ID {obj_id}: {label}", (x1, y1 - 10), config["FONT"],
                        config["LABEL_FONT_SCALE"], (255, 255, 255), config["LABEL_THICKNESS"], cv2.LINE_AA)

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
