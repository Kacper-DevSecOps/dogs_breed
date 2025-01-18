import os
import json
from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import swin_t, Swin_T_Weights
from PIL import Image

app = Flask(__name__)

# Ensure 'static' directory exists
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class labels from JSON
CLASS_LABELS_PATH = "class_labels.json"
if os.path.exists(CLASS_LABELS_PATH):
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
else:
    class_labels = ["Unknown"]  # Fallback if file is missing

num_classes = len(class_labels)

# Load Swin Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, num_classes)

# Load trained weights
MODEL_PATH = "best_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import torch.nn.functional as F  # Import softmax

def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)  # Get raw logits
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        max_prob, predicted = torch.max(probabilities, 1)  # Get max probability & index

    if predicted.item() >= len(class_labels):  # Prevent index error
        return "Unknown Breed", 0.0

    return class_labels[predicted.item()], max_prob.item()  # Return breed & its probability

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        breed, probability = predict_breed(image_path)  # Get breed & probability

        return render_template("index.html", image=image_path, breed=breed, probability=probability)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
