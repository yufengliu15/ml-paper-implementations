from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Link to dataset https://huggingface.co/datasets/ILSVRC/imagenet-1k
ds = load_dataset("imagenet-1k")

# ====================== DATA PREPROCESSING =======================

# Convert numpy array to PIL Image and ensure RGB
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

# ===================== MODEL ARCHITECTURE ==================
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # First convolutional layer 
            nn.Conv2d(
                in_channels=3, # RGB
                out_channels=96, # number of kernels
                kernel_size=11, # (11x11)
                stride=4,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96), #number of channels from conv1
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Second layer
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Third layer
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384), # not specified in paper, but good practice for modern architectures
        
            # Fourth layer
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            # Fifth layer
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # First fully connected layer
            nn.Linear(256 * 6 * 6, 4096), # The spatial dimensions are 6x6 after all the conv and pool layers
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # dropout all hidden neurons that have probability 0.5 to reduce overfitting 

            # Second fully connected layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # Final layer
            nn.Linear(4096, num_classes)
        )
    

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

def testingModel():
    for batch in train_loader:
        images, labels = batch["pixel_values"], batch["label"]
        
        # Move batch to GPU
        images = images.to('cuda')
        labels = labels.to('cuda')
        
        # Forward pass
        outputs = model(images)
        
        # Print shapes to verify everything works
        print(f"Input shape: {images.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Try computing loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f"Loss value: {loss.item()}")
        
        break  # We only need to test one batch

    # Optional: Print model summary
    # print("\nModel Architecture:")
    # print(model)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap train_loader with tqdm for progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        images, labels = batch["pixel_values"], batch["label"]
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print statistics every 100 batches
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss / 100:.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')
            running_loss = 0.0

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total

    return accuracy, avg_loss

def save_checkpoint(model, optimizer, epoch, acc, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['accuracy']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with accuracy {acc:.2f}")
    return epoch, acc, loss

# Create the model
model = AlexNet(num_classes=1000)
model = model.to('cuda')

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005
)

criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 2 # paper uses 90
device = torch.device('cuda')

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    # Train one epoch
    train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Learning rate adjustment (as mentioned in paper)
    if (epoch + 1) % 30 == 0:  # Every 30 epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1  # Reduce learning rate by factor of 10
    
    print(f'Epoch {epoch+1} complete. Training accuracy: {100. * train_acc:.2f}%') 


filename = f'alexnet_checkpoint_epoch_{epoch+1}.pt'
save_checkpoint(model, optimizer, epoch, train_acc, train_loss, filename)