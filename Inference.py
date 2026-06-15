
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Class labels
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def load_model(weights_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model, device


def predict_image(image_path, model, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return CLASS_NAMES[pred.item()]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python Inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    model, device = load_model("landscape_resnet50.pth")

    result = predict_image(image_path, model, device)

    print("Prediction:", result.upper())
