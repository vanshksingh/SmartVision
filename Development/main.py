import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import asyncio
import ssl
import os

# Disable SSL verification (temporary)
ssl._create_default_https_context = ssl._create_unverified_context

# Check if MPS is available and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for training and testing datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
test_dataset = datasets.ImageFolder('dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the classifier to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjust for the number of classes (e.g., 2 classes)

# Move the model to the device (MPS if available)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Asynchronous model saving function
async def async_save_model(epoch, batch_idx):
    save_path = f'banana_tomato_classifier_epoch{epoch}_batch{batch_idx}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Training loop with asyncio for faster processing and frequent saving
async def train_model():
    save_interval = 10  # Save model every 10 batches
    for epoch in range(10):  # 10 epochs
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        print(f"Epoch {epoch + 1}/10")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to the appropriate device (MPS if available)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Print verbose output for every batch
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {total_correct}/{total_samples} ({100.0 * total_correct / total_samples:.2f}%)")

            # Save model asynchronously every `save_interval` batches
            if (batch_idx + 1) % save_interval == 0:
                await async_save_model(epoch, batch_idx + 1)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * total_correct / total_samples
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")

        # Save model at the end of each epoch
        await async_save_model(epoch, batch_idx + 1)

    print("Training complete!")


# Run the training loop with asyncio
asyncio.run(train_model())
