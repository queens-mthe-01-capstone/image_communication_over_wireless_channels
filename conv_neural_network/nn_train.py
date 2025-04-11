
# Import libriaries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt  # For loss visualization
from pathlib import Path

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# List all the image paths and labels
image_paths = []
labels = []

project_root = Path(__file__).resolve().parent.parent

wildfire_image_path = str(project_root / 'images/train/wildfire')
nowildfire_image_path = str(project_root / 'images/train/nowildfire')

wildfire_images = [os.path.join(wildfire_image_path, img)
                   for img in os.listdir(wildfire_image_path) if img.endswith('.jpg') or img.endswith('.jpeg')]
nowildfire_images = [os.path.join(nowildfire_image_path, img)
                     for img in os.listdir(nowildfire_image_path) if img.endswith('.jpg') or img.endswith('.jpeg')]

# Add the paths and labels to the lists
image_paths.extend(wildfire_images)
image_paths.extend(nowildfire_images)
labels.extend([1] * len(wildfire_images))  # 1 for wildfire (fire)
labels.extend([0] * len(nowildfire_images))  # 0 for no wildfire (no fire)

# Create dataset and dataloader
fire_dataset = FireImageDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(fire_dataset, batch_size=16, shuffle=True)
print("Dataset Loaded!")

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

# Initialize the model, loss function, and optimizer
model = FireDetectorCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Split the dataset into train and validation sets
# Instead of a fixed validation dataset, I think this will add more "randoness" to the model in validation
train_size = int(0.8 * len(fire_dataset))  # 80% for training
val_size = len(fire_dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(fire_dataset, [train_size, val_size])

# Create dataloaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training loop with validation
for epoch in range(25):  # Number of epochs
    total_loss = 0
    model.train()  # Set model to training mode

    # Training
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}], Training Loss: {total_loss:.4f}")

    # Validation
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No gradient computation for validation
        for images, labels in val_loader:
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}], Validation Loss: {val_loss:.4f}")

# Plotting the training loss over epochs
plt.plot(range(1, num_epochs+1), train_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

print('Finished Training')

# Save the trained model and optimizer state
model_save_path = "fire_detector_model.pth"
optimizer_save_path = "fire_detector_optimizer.pth"
config_save_path = "training_config.pth"

# Saving the model and optimizer state
torch.save(model.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)

# Saving the training configuration (such as hyperparameters and training loss history)
training_config = {
    'num_epochs': num_epochs,
    'learning_rate': 0.001,
    'batch_size': 16,
    'train_loss': train_loss
}
torch.save(training_config, config_save_path)

print(f"Model and configuration saved to {model_save_path}, {optimizer_save_path}, and {config_save_path}")