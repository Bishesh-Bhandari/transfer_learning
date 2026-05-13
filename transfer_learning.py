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


