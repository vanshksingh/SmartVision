import cv2
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

# List of class names for the YOLO detection
FRUITS = ['banana']  # Include only 'banana' for simplicity

# Define the transform for the ResNet model
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained ResNet model for banana classification
resnet_model = models.resnet18(weights=None)  # Start from scratch (no pre-trained weights)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (ripe/rotten)
resnet_model.load_state_dict(torch.load('banana_tomato_classifier.pth'))  # Load your trained model
resnet_model.eval()  # Set the model to evaluation mode

# Function to predict class and freshness index
def predict_image_with_freshness_index(image, model, transform):
    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Disable gradient calculations for inference
    with torch.no_grad():
        outputs = model(image)  # Get model outputs
        probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get the highest confidence

    # Calculate freshness index (assuming class 0 is 'banana ripe')
    freshness_index = probabilities[0, 0].item() * 10  # Scale to range (0 to 10)

    return predicted.item(), freshness_index

# Function for real-time object detection and classification with YOLO and ResNet
def real_time_detection(model, resnet_model, transform):
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame from the camera

        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO inference on the captured frame
        results = model(frame)

        for result in results:
            boxes = result.boxes  # Detected bounding boxes
            for box in boxes:
                # Get the class index of the detected object
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # Get the name of the detected class

                # Default color for bounding boxes is green
                color = (0, 255, 0)

                # Check if the detected object is a banana
                if class_name.lower() in FRUITS:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers for OpenCV

                    # Crop the detected object from the frame
                    cropped_object = frame[y1:y2, x1:x2]

                    # Convert the cropped image to PIL format
                    cropped_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

                    # Use ResNet model to classify and calculate freshness index
                    class_idx, freshness_index = predict_image_with_freshness_index(cropped_image, resnet_model, transform)

                    # Change the bounding box color to red if the banana is classified as rotten
                    if class_idx == 1:  # Assuming class 1 is 'banana rotten'
                        color = (0, 0, 255)  # Red for rotten bananas

                    # Draw the bounding box around the object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add label and freshness index on the video frame
                    label = f"{class_names[class_idx]}: Freshness {freshness_index:.2f}/10"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                else:
                    # For generic items, draw a green bounding box and label them as 'generic'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = "generic"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for generic
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display the result in a window
        cv2.imshow("YOLO + ResNet Real-Time Detection", frame)

        # Press 'q' to quit the real-time detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load a pre-trained YOLO model (change 'yolov8n.pt' to your model path if needed)
    yolo_model = YOLO("yolo11n.pt")  # You can use any pre-trained YOLO model (YOLOv8n, YOLOv8s, etc.)

    # Define class names for ResNet classification
    class_names = ['banana ripe', 'banana rotten']  # Replace with your class names

    # Start real-time object detection and classification
    real_time_detection(yolo_model, resnet_model, test_transform)
