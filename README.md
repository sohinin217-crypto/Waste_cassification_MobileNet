# 🗑️ Waste Classification — Biodegradable vs Non-Biodegradable

A binary image classifier that takes a photo of a piece of waste and predicts whether it is **biodegradable** or **non-biodegradable**, built with **MobileNetV2** and **TensorFlow/Keras**.

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
MobileNetV2 Backbone (alpha=0.75, ImageNet-pretrained, FROZEN)
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

**Total Parameters**: ~1,383,345 (~1.38 Million)  
**Trainable Parameters**: 1,281 (classification head only)  
**Non-Trainable Parameters**: 1,382,064 (frozen MobileNetV2 backbone)  
**Model Size**: ~5.27 MB  

---

## 📁 Dataset

The dataset is based on the **Recyclable and Household Waste Classification** dataset from Kaggle by [Alistair King](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification).

The 30 original subcategories are remapped into two binary classes:

| Class | Categories |
|-------|-----------|
| **Biodegradable** | cardboard_boxes, cardboard_packaging, coffee_grounds, eggshells, food_waste, tea_bags, magazines, newspaper, office_paper, paper_cups |
| **Non-Biodegradable** | aerosol_cans, aluminum_food_cans, aluminum_soda_cans, steel_food_cans, glass_beverage_bottles, glass_cosmetic_containers, glass_food_jars, plastic_* (9 types), styrofoam_*, clothing, shoes |

- **Total Images**: 15,000 (500 per category × 30 categories)
- **Biodegradable**: 5,000 images
- **Non-Biodegradable**: 10,000 images
- **Split**: 80% Training (12,000) / 20% Validation (3,000)
- **Image Size**: 128×128 RGB

---

## 🚀 Results — Alpha Comparison

| Alpha | Parameters | Model Size | Validation Accuracy |
|-------|-----------|------------|---------------------|
| `alpha=0.35` | 0.41M | ~1.57 MB | 87.33% |
| `alpha=0.5`  | 0.71M | ~2.70 MB | **89.13%** ✅ Best |
| `alpha=0.75` | 1.38M | ~5.27 MB | 88.90% |

> **Best overall**: `alpha=0.5` gives the highest accuracy (89.13%) with a relatively small model (0.71M parameters), making it the best candidate for ESP32 deployment.

### Latest Run (alpha=0.75)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **88.90%** |
| **Validation Loss** | 0.2725 |
| Biodegradable Precision | 0.87 |
| Biodegradable Recall | 0.79 |
| Non-Biodegradable Precision | 0.90 |
| Non-Biodegradable Recall | 0.94 |

### Confusion Matrix (alpha=0.75)
|  | Predicted Bio | Predicted Non-Bio |
|--|:---:|:---:|
| **True Bio** | 804 | 213 |
| **True Non-Bio** | 120 | 1863 |

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

### 3. Prepare the dataset
Download the [Kaggle dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) and place it under:
```
images/
  images/
    aerosol_cans/
    cardboard_boxes/
    ...
```
Then run the preprocessing script:
```bash
python prepare_dataset.py
```
This will create the `dataset/` folder with `biodegradable/` and `non_biodegradable/` subfolders.

### 4. Train the model
```bash
python train.py
```
This will:
- Train for 15 epochs with Adam optimizer (lr=1e-3)
- Save training curves to `training_curves.png`
- Save confusion matrix to `confusion_matrix.png`
- Save sample predictions to `sample_predictions.png`
- Save the trained model to `waste_classifier.keras`

> To change the model width, edit the `alpha` value in `train.py` (line 64). Recommended values: `0.35`, `0.5`, `0.75`, `1.0`.

### 5. Evaluate the saved model
```bash
python evaluate_saved.py
```

---

## 📂 Project Structure

```
├── prepare_dataset.py     # Maps, resizes, and organizes raw images
├── train.py               # Model building, training, and evaluation
├── evaluate_saved.py      # Load and evaluate the saved model
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
Dataset credit: [Alistair King on Kaggle](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
