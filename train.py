import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
import time

# Augment left and right classes by flipping and saving in a new class folder
#from augment_horizontally import augment_images
#source_dir = "./labelled_dataset"
#augmented_left_dir = "./labelled_dataset/right"
#augmented_right_dir = "./labelled_dataset/left"
#augment_images(source_dir, augmented_left_dir, augmented_right_dir)

# Load and modify the pre-trained VGG16 model
print("Loading the VGG model")
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all pretrained layers

print("Adding classifier")
# Update the classifier
input_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(input_features, 256),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Dropout(p=0.8),
    nn.Linear(256, 3)  # Output layer for 3 classes
)

print("Initializing weights")
# Initialize weights for the new layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
model.classifier.apply(init_weights)

print("Setting up hardware acceleration")
# Check for hardware acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

print("Setting up the optimizer and loss function")
#optimizer = optim.AdamW(model.classifier.parameters(), lr=0.0001, weight_decay=1e-4) # 73% test acc
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
class_weights = [.9, .9, .5]
weights = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
print(f"Using class weights: {weights}")

# Setting up data augmentation and data loaders
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset with no transformations initially
dataset = datasets.ImageFolder('./labelled_dataset')

# Show some example transformations
import matplotlib.pyplot as plt
import random
def show_transformed_images(transform, dataset, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    random_indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(random_indices):
        img, label = dataset[idx]  # Get an image and its label
        transformed_img = transform(img)  # Apply the transformations

        # Convert the transformed image back to a displayable format
        transformed_img = transformed_img.permute(1, 2, 0).numpy()  # Rearrange dimensions
        transformed_img = transformed_img - transformed_img.min()  # Normalize for visualization
        transformed_img = transformed_img / transformed_img.max()

        # Display the image
        axes[i].imshow(transformed_img)
        axes[i].set_title(f"Label: {dataset.classes[label]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
show_transformed_images(train_transform, dataset)

# Split indices for training and testing
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply different transformations to the train and testing splits
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# Print dataset statistics
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    average_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return average_loss, accuracy

print("Starting training")
num_epochs = 7
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = 100 * total_correct / total_samples
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)

    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Time: {(time.time() - start_time):.2f}s")

torch.save(model.state_dict(), 'vgg16_model.pth')
print("Model saved as vgg16_model.pth")