import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Fast Local Dataset Paths
    train_dir = '/content/local_dataset/seg_train/seg_train'

    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} not found. Please extract your dataset first.")
        return

    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    # 4. Load Pretrained ResNet-50 & Freeze Base Layers
    print("Loading pretrained ResNet-50...")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer (6 output categories)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 6)
    model = model.to(device)

    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 6. Training Loop (Running 3 quick epochs for verification)
    epochs = 3
    print("Starting model training script...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}% | Time: {end_time-start_time:.2f}s")

    # 7. Save Weights
    torch.save(model.state_dict(), 'landscape_resnet50.pth')
    print("Model training complete. Saved weights to 'landscape_resnet50.pth'")

if __name__ == '__main__':
    train_model()
