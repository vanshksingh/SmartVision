from torchvision import transforms
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F

# Define the transform for a single image (same as used for training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet18(weights=None)  # Start from scratch (no pre-trained weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (adjust to match your number of classes)
model.load_state_dict(torch.load('banana_tomato_classifier.pth'))  # Load trained weights
model.eval()  # Set the model to evaluation mode

# Function to predict class from image and calculate the freshness index
def predict_image_with_freshness_index(image_path, model, transform):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Disable gradient calculations for inference
    with torch.no_grad():
        outputs = model(image)  # Get model outputs
        probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get the highest confidence

    # Calculate freshness index as the probability of being 'banana ripe'
    # Assuming class 0 is 'banana ripe', we will use its probability as a freshness index
    freshness_index = probabilities[0, 0].item() * 10  # Scale to a range (0 to 10)

    return predicted.item(), freshness_index



# Test the model with a new image
image_path = "/Users/vanshkumarsingh/Desktop/IMG_8423.jpeg"  # Replace with the path to your image
class_idx, freshness_index = predict_image_with_freshness_index(image_path, model, test_transform)

# Print the result
class_names = ['banana ripe', 'banana rotten']  # Replace with your class names
print(f"The model predicts: {class_names[class_idx]}")
print(f"Freshness Index: {freshness_index:.2f}/10")
