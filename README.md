# 🗑️ Waste Classification — Biodegradable vs Non-Biodegradable

A binary image classifier that takes a photo of a piece of waste and predicts whether it is **biodegradable** or **non-biodegradable**, built with **MobileNetV2 (alpha=0.5)** and **TensorFlow/Keras**.

This is **Stage 1** of a two-stage project. Stage 2 will convert the trained model to TFLite and deploy it on an **ESP32-S3** microcontroller for real-world waste sorting.

---

## 📊 Model Architecture

```
Input Image (128 x 128 x 3, RGB)
        |
        v
Data Augmentation
  (random flip, rotation, zoom, contrast)
        |
        v
Rescaling [-1, 1] (MobileNetV2 preprocessing)
        |
        v
MobileNetV2 Backbone (alpha=0.5, ImageNet-pretrained, FROZEN)
        |
        v
Global Average Pooling
        |
        v
Dropout (0.2)
        |
        v
Dense(1) + Sigmoid
        |
        v
Output: probability [0.0 = biodegradable | 1.0 = non-biodegradable]
```

**Total Parameters**: ~707,505 (~0.71 Million)  
**Trainable Parameters**: 1,281 (classification head only)  
**Non-Trainable Parameters**: 706,224 (frozen MobileNetV2 backbone)  
**Model Size**: ~2.70 MB  

---

## 📁 Datasets & Evaluation Results

### 1. Main Training Dataset (Kaggle Dataset)
- **Total Images**: 15,000 (5,000 Biodegradable / 10,000 Non-Biodegradable)
- **Validation Split**: 3,000 images (20%)
- **Validation Accuracy**: **89.00%**
- **Validation Loss**: `0.2802`

### 2. Real-World `Datasets` Test Set (DataCluster Garbage Dataset)
- **Total Images**: 250 real-world unsegregated domestic garbage images
- **Validation Accuracy**: **82.40%** (206 / 250 images correct)
- **Validation Loss**: `0.5145`

### 3. Combined Overall Performance

| Dataset | Total Test Images | Accuracy Score |
|:---|:---:|:---:|
| **Main Kaggle Validation Set** | 3,000 | **89.00%** |
| **Real-World `Datasets` Test Set** | 250 | **82.40%** |
| **COMBINED OVERALL TOTAL** | **3,250** | **88.49%** ✅ |

---

## 🛠️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/sohinin217-crypto/Waste_cassification_MobileNet.git
cd Waste_cassification_MobileNet
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\Activate.ps1       # Windows
# source venv/bin/activate       # Linux/macOS

pip install tensorflow pillow matplotlib seaborn scikit-learn tqdm
```

### 3. Predict any single image (Standalone Helper)
```bash
python predict_image.py path/to/waste_photo.jpg
```
Output:
```
==========================================
 Image File  : sample.jpg
 Prediction  : BIODEGRADABLE
 Confidence  : 98.45%
==========================================
```

### 4. Re-evaluate Validation Accuracy
```bash
python evaluate_saved.py              # Evaluates main dataset (89.00%)
python label_and_evaluate_datasets.py # Evaluates Datasets folder (82.40%)
```

---

## 📂 Project Structure

```
├── prepare_dataset.py            # Maps, resizes, and organizes raw images
├── train.py                      # Model building, training, and evaluation
├── evaluate_saved.py             # Evaluates trained model on validation split
├── evaluate_new_dataset.py       # Inference script for unsegregated dataset
├── evaluate_custom_dataset.py    # Custom evaluation on Datasets folder
├── label_and_evaluate_datasets.py# Zero-shot material labeling & accuracy evaluation
├── predict_image.py              # Standalone single-image prediction helper
├── .gitignore
└── README.md
```

> **Note**: The `dataset/`, `images/`, `venv/`, `*.keras`, and `*.png` files are excluded from this repository via `.gitignore` due to their large size.

---

## 🔜 Stage 2 (Coming Soon)
- Convert `waste_classifier.keras` → TFLite (`.tflite`)
- Apply INT8 Post-Training Quantization
- Generate C byte array (`.h` file)
- Deploy on ESP32-S3 using TensorFlow Lite Micro

---

## 📄 License
This project is for educational and research purposes.  
Dataset credits: [Alistair King on Kaggle](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) & DataCluster Labs.
