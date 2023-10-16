import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

LABELS = [
    "ape", "bear", "bison", "cat",
    "chicken", "cow", "deer", "dog",
    "dolphin", "duck", "eagle", "fish",
    "horse", "lion", "lobster", "pig",
    "rabbit", "shark", "snake", "spider",
    "turkey", "wolf"
]

LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat",
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish",
    12: "horse", 13: "lion", 14: "lobster",
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake",
    19: "spider", 20:  "turkey", 21: "wolf"
}

# Define transformations
transform_labeled = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define data paths and parameters
data_path = '/root/code/homework1/Animals Dataset' # you should modify this path
train_and_val_path = os.path.join(data_path, 'train')
batch_size = 32
num_workers = 0

# Create datasets and data loaders
costum_dataset = ImageFolder(train_and_val_path, transform=transform_labeled)
train_size = int(len(costum_dataset) * 0.8)
val_size = len(costum_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(costum_dataset, [train_size, val_size])
