# Brian Lesko
# 6/19/2024
# Use transfer learning to quickly retrain a VGG16 model to classify images.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

print("Loading the VGG model")
# Load and modify the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all the pretrained layers

print("Adding classifier")
# Updating the classifier with the correct input size
input_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(input_features, 256),  # First fully connected layer
    nn.ReLU(),                       # Activation function
    nn.Dropout(p=0.6),               # Dropout for regularization
    nn.Linear(256, 3)                # Output layer for 3 classes
)

print("Initializing classifier weights")
# Initialize weights for the new layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
model.classifier.apply(init_weights)

print("freezing the vgg 16 layers to prevent overfitting")
# Freeze the old layers
for param in model.features.parameters():
    param.requires_grad = False

# Optimizer setup to update only the new classifier layers
print("Setting up the optimizer and loss function")
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.CrossEntropyLoss() # For multi-class classification
print("Using the adam optimizer (adaptive learning rate) and cross entropy loss (for multiclass classification)")


# Check for MPS (Metal Performance Shaders) support
print("Checking for Apple's Metal (MPS) support or CUDA hardware acceleration")
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS if available
elif torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use CUDA if available
else:
    device = torch.device("cpu")  # Fallback to CPU
print(f"Using device: {device}")
model.to(device)

print("Defining the Transformation for input images as the ImageNet standard")
# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class to handle images from two folders
class Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
def extract_label_from_filename(filename):
    label = filename.split('.')[0]  # Take the filename "left.#.jpg"
    return label

class1 = 'left'
class2 = 'right'
class3 = 'straight'
print("Create the test train split")
# Prepare the dataset and dataloaders
photos_dir = './labelled_dataset'
all_images = [(os.path.join(photos_dir, file), extract_label_from_filename(file)) for file in os.listdir(photos_dir)]
# count the number of images in each class
class1_count = len([file for file in os.listdir(photos_dir) if file.startswith(class1)])
class2_count = len([file for file in os.listdir(photos_dir) if file.startswith(class2)])
class3_count = len([file for file in os.listdir(photos_dir) if file.startswith(class3)])
print(f"Class 1 left ({class1}): {class1_count} images")
print(f"Class 2 right ({class2}): {class2_count} images")
print(f"Class 3 straight ({class3}): {class3_count} images")

train_size = round((class1_count + class2_count + class3_count) / 2, 0)
train_images, test_images = train_test_split(all_images, test_size=train_size, train_size=train_size, random_state=42)

train_dataset = Dataset(train_images, transform=transform)
test_dataset = Dataset(test_images, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

def evaluate_model(model, dataloader, criterion, device, tolerance=0.3):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.type(torch.float))
            total_loss += loss.item()
            # Calculate accuracy within tolerance
            predicted = outputs.view(-1)
            total_correct += (torch.abs(predicted - labels) <= tolerance).sum().item()
            total_samples += labels.size(0)
    # Compute average loss and accuracy
    average_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return average_loss, accuracy

print("Starting training")
# Training loop
num_epochs = 7
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.type(torch.float))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs.view(-1) > 0.5).type(torch.float)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}: Loss: {running_loss / (i + 1):.4f}, Accuracy: {correct}/{total} or {accuracy:.2f}%, Time: {(time.time() - start_time):.2f} seconds")

#### Save and load the model
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
print("Model loaded from model.pth")