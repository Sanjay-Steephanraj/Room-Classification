
# 🏨 AI Hotel Room Classification System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![Flask](https://img.shields.io/badge/Flask-WebApp-black.svg)
![Tailwind](https://img.shields.io/badge/TailwindCSS-Frontend-38B2AC.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

An **AI-powered hotel room image classification system** that automatically categorizes hotel room images into different room types using a **ConvNeXt deep learning model**.

The project includes:

- 🧠 A **deep learning training pipeline**
- ⚡ A **Flask inference server**
- 🌐 A **modern dashboard UI**
- 📊 Visual analytics of classification results

This system helps **hotel platforms, property managers, and ML researchers automatically organize property images into structured categories.**

---

# 🚀 Key Features

### 🧠 AI-Based Image Classification
Uses **ConvNeXt Tiny architecture** trained using PyTorch to classify hotel images.

### 📂 Multiple Input Options
The system supports:

- Local folder image classification
- CSV upload containing image URLs

### 🌐 Interactive Web Dashboard
Users can:

- Upload image sources
- View classification results
- Analyze grouped outputs visually

### 📊 Smart Analytics Panel

Displays:

- Number of images per category
- Average confidence per class
- Visual grouping of results

### ⚡ High Performance

- GPU acceleration supported
- Efficient image preprocessing pipeline
- Lightweight model inference

---

# 🏗 System Architecture

```
        User Input
       /          \
  Local Folder     CSV URLs
       |              |
       ------ Flask Backend ------
                    |
            Image Preprocessing
                    |
              ConvNeXt Model
                    |
             Prediction Engine
                    |
           JSON Classification
                    |
           Dashboard Visualization
```

---

# 🧠 Model Information

The classification model is built using:

**ConvNeXt Tiny** (from the TIMM library)

Deep learning framework:

- PyTorch
- Torchvision
- TIMM

### Image Preprocessing

Images are transformed using:

- Resize → 236
- CenterCrop → 224
- Normalize → ImageNet mean & std

### Prediction Rule

If confidence < **50%**, the image is classified as:

```
Others
```

---

# 🏷 Supported Categories

The model predicts **14 hotel room categories**:

- BBQ
- Bathroom
- Bedroom
- Common Space
- Dining Room
- Exterior
- Gym Room
- Kitchen
- Living Room
- Lounge
- Play Area
- Study Room
- Swimming Pool
- View

---

# 📂 Project Structure

```
AI-Hotel-Room-Classifier
│
├── hotel_room_classifier_training_setup.ipynb
│       Training pipeline notebook
│
├── best_convnext_hotel_classifier.pth
│       Trained model weights
│
├── app.py
│       Flask backend server
│
├── templates
│      └── index.html
│           Web dashboard UI
│
└── README.md
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/hotel-room-classifier.git
cd hotel-room-classifier
```

---

## 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install torch torchvision timm flask pillow requests
```

---

# ▶️ Running the Application

Start the Flask server:

```
python app.py
```

The application will run at:

```
http://127.0.0.1:5000
```

Open it in your browser.

---

# 📂 How to Use

## Option 1 — Local Folder

Enter a folder path in the dashboard:

```
C:/images/test
```

The system will:

1. Load all images
2. Classify them
3. Display grouped results

---

## Option 2 — CSV Upload

Upload a CSV containing image URLs.

Example CSV:

```
https://example.com/image1.jpg
https://example.com/image2.jpg
https://example.com/image3.jpg
```

The system will:

1. Download images
2. Run AI classification
3. Display categorized results

---

# 📊 Dashboard Output

The dashboard displays:

### Category Statistics

- Total images per category
- Average confidence score

### Visual Results

Each image shows:

```
Category + Confidence
```

Example:

```
Bedroom 92.4%
Bathroom 87.3%
Others 45.1%
```

### Confidence Color Indicators

| Confidence | Color |
|-----------|------|
| >70% | Green |
| 50–70% | Yellow |
| <50% | Red |

---

# 🧪 Training Pipeline

Training notebook:

```
hotel_room_classifier_training_setup.ipynb
```

Pipeline includes:

- Dataset preparation
- Transfer learning with ConvNeXt
- Model training
- Validation
- Checkpoint saving

---

# 🔮 Future Improvements

Potential upgrades:

- Drag & drop image upload
- Real-time webcam classification
- Automatic dataset expansion
- Cloud deployment
- Docker container support
- REST API documentation

---

# 👨‍💻 Tech Stack

### Backend

- Flask
- Python
- Requests

### Deep Learning

- PyTorch
- Torchvision
- TIMM
- ConvNeXt

### Frontend

- HTML
- TailwindCSS
- JavaScript

---

# 📜 License

This project is created for **educational and research purposes**.

---

# ⭐ If you like this project

Consider giving the repository a **star ⭐ on GitHub** to support the work!
