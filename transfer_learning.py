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