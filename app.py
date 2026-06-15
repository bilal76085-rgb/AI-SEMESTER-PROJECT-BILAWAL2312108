import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
WEIGHTS_PATH = 'landscape_resnet50.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError("Model weights not found!")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return jsonify({
        'prediction': CLASS_NAMES[pred.item()].upper()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)