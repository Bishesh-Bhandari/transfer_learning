import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import numpy as np
import random
import time
import copy

from pathlib import Path


# Device selection: prioritize GPU acceleration when available

def get_device():

    """

    Automatically select the best available device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

        print("Using MPS (Apple Silicon GPU)")

    else:

        device = torch.device("cpu")

        print("Using CPU")

    return device

 

device = get_device()


def set_seed(seed=42):

    """Set random seeds for reproducibility across all libraries."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may slow down training)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

 

set_seed(42)

# ImageNet normalization values (used for all pretrained models)

# Mean and SD of pixel values across all ImageNet images for each color channel (R, G, B)

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

 

# Transforms for training data (with augmentation)

train_transforms = transforms.Compose([

    transforms.Resize(224),              # Resize to 224x224 for pretrained models

    transforms.RandomHorizontalFlip(),   # Simple augmentation

    transforms.ToTensor(),               # Convert to tensor [0, 1]

    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # ImageNet normalization

])

 

# Transforms for validation/test data (no augmentation)

val_transforms = transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

])


# Download and prepare datasets

print("Downloading CIFAR-10 dataset...")

train_dataset = datasets.CIFAR10(

    root='./data',

    train=True,

    download=True,

    transform=train_transforms

)

 

val_dataset = datasets.CIFAR10(

    root='./data',

    train=False,

    download=True,

    transform=val_transforms

)

 

# CIFAR-10 class names

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(class_names)

 

print(f"Training samples: {len(train_dataset)}")

print(f"Validation samples: {len(val_dataset)}")

print(f"Number of classes: {num_classes}")

BATCH_SIZE = 32

 

# Note: num_workers=0 for Windows compatibility; increase on Linux/Mac for speed

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,           # Shuffle training data each epoch

    num_workers=0,

    pin_memory=True if device.type == 'cuda' else False

)

 

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,          # No need to shuffle validation data

    num_workers=0,

    pin_memory=True if device.type == 'cuda' else False

)

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train

        dataloader: Training data loader

        criterion: Loss function

        optimizer: Optimizer (e.g., Adam, SGD)

        device: Device to use (cuda/mps/cpu)

    Returns:
        Tuple of (average_loss, accuracy)

    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)

    running_loss = 0.0

    correct = 0

    total = 0

    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()           # Clear previous gradients

        outputs = model(inputs)         # Forward pass

        loss = criterion(outputs, labels)

        loss.backward()                 # Compute gradients

        optimizer.step()                # Update weights

        

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    

    epoch_loss = running_loss / total

    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc



def evaluate(model, dataloader, criterion, device):

    """
    Evaluate the model on validation/test data.

    Args:
        model: PyTorch model to evaluate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, accuracy)

    """
    model.eval()  # Set model to evaluation mode (disables dropout, fixes batch norm)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / total

    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc



def train_model(model, train_loader, val_loader, criterion, optimizer, 

                device, num_epochs, scheduler=None, model_name="Model"):

    """

    Complete training loop with validation and history tracking.

    

    Returns:

        Tuple of (trained_model, history_dict)

    """

    # History dictionary to store loss and accuracy for each epoch

    history = {

        'train_loss': [], 'train_acc': [],

        'val_loss': [], 'val_acc': []

    }

    

    best_acc = 0.0

    best_model_weights = copy.deepcopy(model.state_dict())

    

    print(f"\n{'='*60}")

    print(f"Training: {model_name}")

    print(f"{'='*60}")

    

    start_time = time.time()

    for epoch in range(num_epochs):

        epoch_start = time.time()

        # Training phase

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step()

# Save history

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_acc:

            best_acc = val_acc

            best_model_weights = copy.deepcopy(model.state_dict())

        

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "

              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "

              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "

              f"Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time

    print(f"\nTraining complete in {total_time/60:.1f} minutes")

    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, history


def plot_curves(history, title="Training History"):

    """

    Plot training and validation loss/accuracy curves.

    

    Args:

        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'

        title: Plot title

    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    

    # Loss plot

    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')

    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')

    axes[0].set_xlabel('Epoch')

    axes[0].set_ylabel('Loss')

    axes[0].set_title(f'{title} - Loss')

    axes[0].legend()

    axes[0].grid(True)

    

    # Accuracy plot

    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')

    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')

    axes[1].set_xlabel('Epoch')

    axes[1].set_ylabel('Accuracy (%)')

    axes[1].set_title(f'{title} - Accuracy')

    axes[1].legend()

    axes[1].grid(True)

    

    plt.tight_layout()

    plt.show()

def show_predictions(model, dataloader, class_names, device, num_images=8):

    """
    Display sample images with predicted and true labels.
    """

    model.eval()

    images_shown = 0

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    axes = axes.flatten()

    

    # Denormalize for display

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)

    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = outputs.max(1)

            

            for i in range(inputs.size(0)):

                if images_shown >= num_images:

                    break

                

                # Denormalize image

                img = inputs[i].cpu() * std + mean

                img = img.permute(1, 2, 0).numpy()

                img = np.clip(img, 0, 1)

                true_label = class_names[labels[i]]

                pred_label = class_names[predicted[i]]

                color = 'green' if true_label == pred_label else 'red'

                

                axes[images_shown].imshow(img)

                axes[images_shown].set_title(f'True: {true_label}\nPred: {pred_label}', 

                                             color=color, fontsize=10)

                axes[images_shown].axis('off')

                images_shown += 1

            

            if images_shown >= num_images:

                break

    

    plt.tight_layout()

    plt.show()

# Load pretrained ResNet18

# weights=IMAGENET1K_V1 loads the ImageNet pretrained weights

feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

 

# Examine the model structure (helps understand what to modify)

print("Original ResNet18 final layer (fc):")

print(feature_extractor.fc)

# STEP 2: Replace the final fully connected layer

# ResNet18's fc layer expects 512 input features, outputs 1000 (ImageNet classes)

# We replace it with a new layer that outputs num_classes (10 for CIFAR-10)

num_features = feature_extractor.fc.in_features  # Get input size (512 for ResNet18)

feature_extractor.fc = nn.Linear(num_features, num_classes)  # New layer, trainable by default

 

print(f"New final layer: Linear({num_features}, {num_classes})")


# Move model to device

feature_extractor = feature_extractor.to(device)

 

# Count trainable vs total parameters

total_params = sum(p.numel() for p in feature_extractor.parameters())

trainable_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")

print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

# STEP 3: Set up optimizer - only optimize the new head (fc layer)

# Since other layers are frozen, we only need to pass the trainable parameters

criterion = nn.CrossEntropyLoss()

optimizer_fe = optim.Adam(feature_extractor.fc.parameters(), lr=0.001)

 

# Train the feature extractor

FEATURE_EXTRACTOR_EPOCHS = 5

 

feature_extractor, fe_history = train_model(

    model=feature_extractor,

    train_loader=train_loader,

    val_loader=val_loader,

    criterion=criterion,

    optimizer=optimizer_fe,

    device=device,

    num_epochs=FEATURE_EXTRACTOR_EPOCHS,

    model_name="ResNet18 Feature Extractor (Frozen)"

)

