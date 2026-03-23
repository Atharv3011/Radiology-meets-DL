import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS


# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve model file paths relative to this file so they load regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "binary_fracture_mura_gpu.pth")
MULTICLASS_MODEL_PATH = os.path.join(BASE_DIR, "multiclass_fracture_gpu.pth")

MULTICLASS_CLASSES = [
    "Compression-Crush fracture",
    "Fracture Dislocation",
    "Hairline Fracture",
    "Impacted fracture",
    "Longitudinal fracture",
    "Oblique fracture",
    "Spiral Fracture"
]

# ==========================================================
# MODEL DEFINITIONS
# ==========================================================
def load_binary_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    # Match the trained binary head: 2048 -> 256 -> 2
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model



def load_multiclass_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(MULTICLASS_CLASSES))
    )
    model.load_state_dict(torch.load(MULTICLASS_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ==========================================================
# LOAD MODELS
# ==========================================================
print(f"Using device: {DEVICE}")
binary_model = load_binary_model()
multiclass_model = load_multiclass_model()

# ==========================================================
# FLASK APP
# ==========================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Step 1: Binary classification
    with torch.no_grad():
        binary_out = torch.softmax(binary_model(image), dim=1)
        binary_conf, binary_pred = torch.max(binary_out, 1)
        fracture_detected = binary_pred.item() == 1  # Assuming class 1 = Fracture

    if not fracture_detected:
        return jsonify({
            "fracture_detected": False,
            "label": "No Fracture Detected",
            "confidence": round(binary_conf.item() * 100, 2)
        })

    # Step 2: Multiclass classification
    with torch.no_grad():
        multi_out = torch.softmax(multiclass_model(image), dim=1)
        conf, pred = torch.max(multi_out, 1)
        predicted_label = MULTICLASS_CLASSES[pred.item()]

    return jsonify({
        "fracture_detected": True,
        "fracture_type": predicted_label,
        "confidence": round(conf.item() * 100, 2)
    })

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

