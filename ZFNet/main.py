from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ds = load_dataset("imagenet-1k")

# DATA PREPROCESSING
def to_rgb_pil(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Lambda(to_rgb_pil),  # Convert to PIL and ensure RGB
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def transform_dataset(examples):
    # Transform images
    images = [transform(image) for image in examples["image"]]
    # Stack them into a tensor
    images = torch.stack(images)
    return {"pixel_values": images, "label": examples["label"]}

train_ds = ds["train"].with_transform(transform_dataset)

train_loader = DataLoader(
    train_ds,
    batch_size=128, 
    shuffle=True,    # Randomly shuffle data
    num_workers=4    # Parallel data loading
)