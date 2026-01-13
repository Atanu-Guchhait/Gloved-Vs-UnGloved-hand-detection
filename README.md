# 🧤 Glove vs Bare Hand Detection System (YOLOv8)

A complete **end-to-end object detection pipeline** to identify whether a person is wearing **safety gloves** or has a **bare (uncovered) hand**, designed for **industrial safety compliance** use cases.

This project demonstrates **real-world computer vision workflow**: dataset handling, model training, inference, result visualization, and structured logging.

---

## 📌 Problem Statement

In industrial environments, wearing gloves is often mandatory for safety compliance.  
This system automatically detects:

- **gloved_hand** → Hand wearing protective gloves  
- **bare_hand** → Normal hand without gloves  

from images captured by factory cameras.

---

## 🚀 Features

- Custom **YOLOv8 object detection model**
- Trained on real-world labeled dataset
- Batch inference on folders of images
- Annotated image outputs with bounding boxes
- Detection results saved in **structured JSON format**
- CLI-based execution (production-style)
- Easily extendable to video / CCTV streams

---

## 🛠️ Tech Stack

- **Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Language**: Python 3.x
- **Libraries**:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `tqdm`

```text
GloveVSUnGlove/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/
│       ├── images/
│       └── labels/
│
├── input_images/            # Images used for inference
│
├── output/                  # Output results
│   ├── images/              # Annotated images
│   └── detections.json      # Detection results in JSON format
│
├── data.yml                 # Dataset configuration file
├── train.py                 # Model training script
├── detect_folder.py         # Folder-based inference + JSON logging
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```


## 📊 Dataset

- Dataset format: **YOLO annotation format**
- Labels:
  - `0` → gloved_hand
  - `1` → bare_hand
- Each image has a corresponding `.txt` file with normalized bounding boxes

Example label file:


0 0.52 0.48 0.30 0.35


---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv myenv
myenv\Scripts\activate   # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

🧠 Training the Model
Dataset Configuration (data.yml)
path: dataset

train: train/images
val: valid/images
test: test/images

names:
  0: gloved_hand
  1: bare_hand

Train Command
python train.py


After training, the best model is saved at:

runs/detect/glove_detector/weights/best.pt

🔍 Inference on Image Folder
Run Detection
python detect_folder.py --input input_images --output output --confidence 0.5

Output Generated
output/
├── images/          # Annotated images
└── detections.json  # Detection logs

📄 JSON Output Format
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [120, 80, 300, 260]
    }
  ]
}

📈 Model Performance (Example)

Gloved Hand Detection: High precision

Bare Hand Detection: Moderate precision

Accuracy improves with:

More training epochs

Larger YOLO models (YOLOv8m)

Better bounding box quality

🧪 Observations & Challenges

Visual similarity between gloves and skin can cause confusion

Large or loose bounding boxes reduce accuracy

Dataset quality directly impacts performance

Multiple training iterations are required for real-world robustness

🧠 Key Learnings

End-to-end CV pipeline design

Object detection model fine-tuning

Dataset debugging and validation

Practical deployment considerations

CLI-based ML tooling (industry standard)

🔮 Future Improvements

🎥 Video / webcam detection

🌐 Streamlit web dashboard

⚖️ Class balancing techniques

📊 Metrics visualization

☁️ Cloud deployment

🏭 Real-World Applications

Factory safety compliance monitoring

PPE detection systems

Workplace automation

Industrial computer vision solutions

📬 Author

Atanu Guchhait
B.Tech (CSE – AI & ML)
Aspiring Data Scientist / ML Engineer
## 📁 Project Structure

