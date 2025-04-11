
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# CNN model for image classification
class FireDetectorCNN(nn.Module):
    def __init__(self):
        super(FireDetectorCNN, self).__init__()

        # Define a simple CNN with 2D convolutions for image classification
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Input 1 channel for grayscale images
        self.a1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.a2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.a3 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling with 2x2 kernel
        self.fc1 = nn.Linear(256 * 32 * 32, 1024)  # Flatten image to 1024 features
        self.a4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1)  # Output layer for binary classification (fire/no fire)
        self.sigmoid = nn.Sigmoid()  # Output probability between 0 and 1

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.pool(self.a1(self.conv1(x)))
        x = self.pool(self.a2(self.conv2(x)))
        x = self.pool(self.a3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers

        x = self.dropout(self.a4(self.fc1(x)))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Output probability of fire (binary classification)

        return x

# Define transformations for the images (resize, convert to grayscale, convert to tensor, and normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize image
])

# Dataset class for loading fire images (without directories)
class FireImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # List of image paths
        self.labels = labels  # Corresponding labels
        self.transform = transform  # Transformation to apply to each image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Load image and convert to grayscale (L mode)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Load test image paths and labels
test_image_paths = []
test_labels = []

project_root = Path(__file__).resolve().parent.parent

wildfire_image_path = str(project_root / 'images/train/wildfire')
nowildfire_image_path = str(project_root / 'images/train/nowildfire')

wildfire_test_images = [os.path.join(wildfire_image_path, img)
                        for img in os.listdir(wildfire_image_path) if img.endswith('.jpg') or img.endswith('.jpeg')]
nowildfire_test_images = [os.path.join(nowildfire_image_path, img)
                          for img in os.listdir(nowildfire_image_path) if img.endswith('.jpg') or img.endswith('.jpeg')]

test_image_paths.extend(wildfire_test_images)
test_image_paths.extend(nowildfire_test_images)
test_labels.extend([1] * len(wildfire_test_images))  # Label 1 for wildfire
test_labels.extend([0] * len(nowildfire_test_images))  # Label 0 for no wildfire

# Create test dataset and dataloader
test_dataset = FireImageDataset(test_image_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the model
model = FireDetectorCNN()
model.load_state_dict(torch.load('fire_detector_model.pth'))
model.eval()  # Set model to evaluation mode

def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in dataloader:
            outputs = model(images).squeeze()
            preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)



    # Convert lists to numpy arrays for confusion matrix calculation
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting the confusion matrix using seaborn's heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=['No Wildfire', 'Wildfire'], yticklabels=['No Wildfire', 'Wildfire'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, f1

# Run evaluation
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")