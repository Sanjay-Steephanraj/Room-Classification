import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import requests
from io import BytesIO, TextIOWrapper
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# ==============================
# Model
# ==============================
class HotelRoomClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


NUM_CLASSES = 14
MODEL_PATH = "best_convnext_hotel_classifier.pth"
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HotelRoomClassifier(NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
print("MODEL LOADED")

# ==============================
# Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = [
    "BBQ", "Bathroom", "Bedroom", "Common Space", "Dining Room", "Exterior",
    "Gym room", "Kitchen", "Living Room", "Lounge", "Play Area",
    "Study Room", "Swimming pool", "View"
]

# ==============================
# Prediction
# ==============================
def predict_pil(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, 1)
        conf, pred = torch.max(probs, 1)

    conf = conf.item()
    label = class_names[pred.item()] if conf >= THRESHOLD else "Others"

    return label, round(conf * 100, 2)

# ==============================
# Main Route
# ==============================
@app.route("/predict_folder", methods=["POST"])
def predict_folder():
    try:
        results = []

        # =============================
        # CSV Upload
        # =============================
        if request.files.get("file"):

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "Empty CSV file"}), 400

            csv_file = TextIOWrapper(file.stream, encoding="utf-8")
            reader = csv.reader(csv_file)

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            for row in reader:
                if not row:
                    continue

                url = row[0].strip()

                # Skip header if present
                if url.lower().startswith("http") is False:
                    continue

                try:
                    print("Processing:", url)

                    response = requests.get(url, headers=headers, timeout=10)

                    if response.status_code != 200:
                        print("Bad status:", response.status_code)
                        continue

                    content_type = response.headers.get("Content-Type", "")

                    if "image" not in content_type:
                        print("Not an image URL:", url)
                        continue

                    image = Image.open(BytesIO(response.content)).convert("RGB")

                    label, conf = predict_pil(image)

                    results.append({
                        "image": url,
                        "folder": "CSV",
                        "class": label,
                        "confidence": conf,
                        "is_url": True
                    })

                except Exception as e:
                    print("CSV image error:", e)
                    continue

            return jsonify(results)

        # =============================
        # Folder JSON
        # =============================
        elif request.is_json:

            data = request.get_json()

            if not data or "folder_path" not in data:
                return jsonify({"error": "No folder path provided"}), 400

            input_path = data["folder_path"]

            if os.path.exists(input_path):

                for file in os.listdir(input_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_path = os.path.join(input_path, file)
                        image = Image.open(img_path).convert("RGB")

                        label, conf = predict_pil(image)

                        results.append({
                            "image": file,
                            "folder": input_path,
                            "class": label,
                            "confidence": conf,
                            "is_url": False
                        })

                return jsonify(results)

            else:
                return jsonify({"error": "Folder not found"}), 400

        else:
            return jsonify({"error": "Unsupported request type"}), 415

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ==============================
# Serve Local Images
# ==============================
@app.route("/get_image")
def get_image():
    folder = request.args.get("folder")
    filename = request.args.get("filename")
    return send_from_directory(folder, filename)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)