import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Paths
datasets_dir = r"d:\Dekstop\archive (3)\Datasets"
model_path = r"d:\Dekstop\archive (3)\waste_classifier.keras"
output_plot_path = r"d:\Dekstop\archive (3)\custom_datasets_predictions.png"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

if not os.path.exists(datasets_dir):
    print(f"Error: Directory not found at {datasets_dir}")
    print("Please place your 'Datasets' folder at d:\\Dekstop\\archive (3)\\Datasets and try again.")
    exit(1)

print(f"Found 'Datasets' folder at {datasets_dir}")

# Check subfolder structure
subfolders = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
print(f"Subfolders detected in Datasets: {subfolders}")

# Build MobileNetV2 (alpha=0.5) model architecture
print("Building model architecture (alpha=0.5)...")
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    alpha=0.5,
    include_top=False,
    weights=None
)
backbone.trainable = False

inputs = tf.keras.layers.Input(shape=(128, 128, 3))
x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0)(inputs)
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs, outputs, name="waste_classifier")

print(f"Loading weights from {model_path}...")
model.load_weights(model_path)

# Check if dataset has binary subfolders (biodegradable vs non_biodegradable)
has_binary = any(name in [s.lower() for s in subfolders] for name in ['biodegradable', 'non_biodegradable'])

if has_binary:
    print("\nDetected labeled dataset structure. Evaluating accuracy...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        datasets_dir,
        image_size=(128, 128),
        batch_size=32,
        label_mode='binary',
        shuffle=False
    )
    
    class_names = val_ds.class_names
    print(f"Classes: {class_names}")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    loss, accuracy = model.evaluate(val_ds)
    print(f"\n==========================================")
    print(f" Validation Accuracy on Datasets: {accuracy * 100:.2f}%")
    print(f" Validation Loss on Datasets    : {loss:.4f}")
    print(f"==========================================\n")
    
    # Detailed report
    all_true = []
    all_preds = []
    for imgs, lbls in val_ds:
        p = model.predict(imgs, verbose=0)
        all_true.extend(lbls.numpy().flatten())
        all_preds.extend(p.flatten())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_pred_lbls = (all_preds >= 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(all_true, all_pred_lbls, target_names=class_names))
    
else:
    print("\nDataset does not contain 'biodegradable'/'non_biodegradable' subfolders.")
    print("Running inference on all images in 'Datasets'...")
    
    image_files = sorted(glob.glob(os.path.join(datasets_dir, "**", "*.jpg"), recursive=True) + 
                          glob.glob(os.path.join(datasets_dir, "**", "*.png"), recursive=True) +
                          glob.glob(os.path.join(datasets_dir, "**", "*.jpeg"), recursive=True))
    
    print(f"Found {len(image_files)} image files.")
    processed_images = []
    valid_files = []
    
    for fpath in image_files:
        try:
            with Image.open(fpath) as img:
                rgb_img = img.convert("RGB")
                resized_img = rgb_img.resize((128, 128), Image.Resampling.LANCZOS)
                processed_images.append(np.array(resized_img, dtype=np.float32))
                valid_files.append(fpath)
        except Exception as e:
            pass
            
    images_tensor = np.array(processed_images)
    preds = model.predict(images_tensor, batch_size=32, verbose=1)
    
    bio = sum(1 for p in preds if p[0] < 0.5)
    non_bio = len(preds) - bio
    print(f"\nEvaluated {len(preds)} images:")
    print(f"  Biodegradable    : {bio} ({bio/len(preds):.2%})")
    print(f"  Non-Biodegradable: {non_bio} ({non_bio/len(preds):.2%})")
