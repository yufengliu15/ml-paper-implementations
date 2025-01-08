from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ds = load_dataset("imagenet-1k")
val_ds = ds["validation"]  

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

def to_rgb_pil(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Lambda(to_rgb_pil),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():  # No need to track gradients
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, labels = batch["pixel_values"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_predicted = outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(top5_predicted)
            correct = top5_predicted.eq(labels).float()
            top5_correct += correct[:, :5].sum().item()
            
            total += labels.size(0)
    
    return (top1_correct / total) * 100, (top5_correct / total) * 100

def display_prediction(model, dataset, class_names=None):
    # Set model to evaluation mode
    model.eval()
    
    # Randomly select an image
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]
    
    # Get original image for display (before transformation)
    original_image = to_rgb_pil(sample['image'])
    
    # Transform image for model
    transformed_image = transform(original_image)
    # Add batch dimension
    transformed_image = transformed_image.unsqueeze(0).to('cuda')
    
    # Get predictions
    with torch.no_grad():
        outputs = model(transformed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
    # Convert probabilities and indices to lists
    top5_prob = top5_prob[0].cpu().numpy() * 100
    top5_idx = top5_idx[0].cpu().numpy()
    
    # Display the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(5)
    plt.barh(y_pos, top5_prob)
    plt.yticks(y_pos, [f"Class {idx}" if class_names is None else class_names[idx] for idx in top5_idx])
    plt.xlabel('Probability (%)')
    plt.title('Top 5 Predictions')
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("\nTop 5 predictions:")
    for prob, idx in zip(top5_prob, top5_idx):
        class_name = f"Class {idx}" if class_names is None else class_names[idx]
        print(f"{class_name}: {prob:.2f}%")

def main():
    # Load your model architecture
    model = AlexNet(num_classes=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load your saved checkpoint
    checkpoint = torch.load('alexnet_checkpoint_epoch_2.pt')  # adjust filename as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare validation dataset
    def transform_dataset(examples):
        images = [transform(image) for image in examples["image"]]
        images = torch.stack(images)
        return {"pixel_values": images, "label": examples["label"]}

    val_ds = ds["validation"].with_transform(transform_dataset)
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,  # No need to shuffle for validation
        num_workers=4
    )
    
    # Evaluate
    top1_acc, top5_acc = evaluate_model(model, val_loader, device)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    # Randomly sample 1 image and display results
    model = AlexNet(num_classes=1000)
    model = model.to('cuda')

    checkpoint = torch.load('alexnet_checkpoint_epoch_2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    ds = load_dataset("imagenet-1k")

    class_names = ds['train'].features['label'].names

    display_prediction(model, ds['validation'], class_names)