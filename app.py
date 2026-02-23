import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import requests
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image
from urllib.parse import urljoin

app = Flask(__name__)

# ==============================
# Model
# ==============================
class HotelRoomClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

NUM_CLASSES = 14
MODEL_PATH = "convnext_model.pth"
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
# CORRECT ConvNeXt Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_names = [
    "BBQ","Bathroom","Bedroom","Common Space","Dining Room","Exterior",
    "Gym room","Kitchen","Living Room","Lounge","Play Area","Study Room",
    "Swimming pool","View"
]

# ==============================
# Predict Single Image
# ==============================
def predict_pil(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs,1)
        conf, pred = torch.max(probs,1)

    conf = conf.item()
    label = class_names[pred.item()] if conf >= THRESHOLD else "Others"
    return label, round(conf*100,2)

# ==============================
# CSV URL PROCESSOR
# ==============================
def process_csv(csv_path):
    results = []

    # local csv
    if os.path.exists(csv_path):
        f = open(csv_path, newline='', encoding='utf-8')
        reader = csv.reader(f)
        urls = [row[0] for row in reader]

    # url csv
    else:
        response = requests.get(csv_path)
        urls = response.text.splitlines()

    for url in urls:
        try:
            img = Image.open(BytesIO(requests.get(url,timeout=10).content)).convert("RGB")
            label, conf = predict_pil(img)

            results.append({
                "image": url,
                "folder": "CSV_URL",
                "class": label,
                "confidence": conf
            })
        except:
            continue

    return results

# ==============================
# Folder Processor
# ==============================
def process_folder(folder_path):
    results=[]
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg",".png",".jpeg",".webp")):
            img_path=os.path.join(folder_path,file)
            img=Image.open(img_path).convert("RGB")
            label,conf=predict_pil(img)

            results.append({
                "image":file,
                "folder":folder_path,
                "class":label,
                "confidence":conf
            })
    return results

# ==============================
# MAIN ROUTE
# ==============================
@app.route("/predict_folder", methods=["POST"])
def predict_folder():
    try:
        results = []

        # =====================================
        # CASE 1: CSV Upload (multipart/form)
        # =====================================
        if request.files.get("file"):

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "Empty CSV file"}), 400

            import csv
            from io import TextIOWrapper

            csv_file = TextIOWrapper(file.stream, encoding="utf-8")
            reader = csv.reader(csv_file)

            for row in reader:
                if not row:
                    continue

                url = row[0]

                try:
                    response = requests.get(url, timeout=10)
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

        # =====================================
        # CASE 2: JSON (Folder or Website URL)
        # =====================================
        elif request.is_json:

            data = request.get_json()

            if not data or "folder_path" not in data:
                return jsonify({"error": "No folder path provided"}), 400

            input_path = data["folder_path"]

            # -------- Local Folder --------
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
    folder=request.args.get("folder")
    filename=request.args.get("filename")
    return send_from_directory(folder,filename)

@app.route("/")
def home():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
