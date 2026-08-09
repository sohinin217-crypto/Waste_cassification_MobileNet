import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Paths
model_path = r"d:\Dekstop\archive (3)\waste_classifier.keras"
new_images_dir = r"d:\Dekstop\archive (3)\data\archive\trash\trash"
output_plot_path = r"d:\Dekstop\archive (3)\new_dataset_predictions.png"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

if not os.path.exists(new_images_dir):
    print(f"Error: Image directory not found at {new_images_dir}")
    exit(1)

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

# Class names corresponding to training alphabetical order:
# index 0: biodegradable, index 1: non_biodegradable
class_names = ['biodegradable', 'non_biodegradable']

# Gather image files
image_files = sorted(glob.glob(os.path.join(new_images_dir, "*.jpg")) + 
                      glob.glob(os.path.join(new_images_dir, "*.png")) +
                      glob.glob(os.path.join(new_images_dir, "*.jpeg")))

print(f"Found {len(image_files)} images in the new dataset.")

processed_images = []
valid_files = []

for file_path in image_files:
    try:
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize((128, 128), Image.Resampling.LANCZOS)
            img_arr = np.array(resized_img, dtype=np.float32)
            processed_images.append(img_arr)
            valid_files.append(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

images_tensor = np.array(processed_images)

print(f"Running inference on {len(images_tensor)} images...")
preds = model.predict(images_tensor, batch_size=32, verbose=1)

bio_count = 0
non_bio_count = 0

results = []

print("\n--- Predictions on New Dataset ---")
print(f"{'Index':<6} | {'Image File':<35} | {'Predicted Class':<18} | {'Confidence':<10}")
print("-" * 75)

for i in range(len(preds)):
    prob = float(preds[i][0])
    pred_class_idx = 1 if prob >= 0.5 else 0
    pred_label = class_names[pred_class_idx]
    conf = prob if pred_class_idx == 1 else (1.0 - prob)
    
    file_name = os.path.basename(valid_files[i])
    results.append((file_name, pred_label, conf, processed_images[i]))
    
    if pred_class_idx == 0:
        bio_count += 1
    else:
        non_bio_count += 1
        
    if i < 25:  # Print first 25 predictions to console
        print(f"{i+1:<6} | {file_name:<35} | {pred_label:<18} | {conf:.2%}")

if len(preds) > 25:
    print(f"... and {len(preds) - 25} more images.")

print("-" * 75)
print("\n--- Overall Class Distribution on New Dataset ---")
print(f"Total Evaluated Images: {len(preds)}")
print(f"Predicted Biodegradable    : {bio_count} ({bio_count/len(preds):.2%})")
print(f"Predicted Non-Biodegradable: {non_bio_count} ({non_bio_count/len(preds):.2%})")

# Plot 16 sample predictions
plt.figure(figsize=(14, 14))
sample_indices = np.linspace(0, len(results) - 1, min(16, len(results)), dtype=int)

for idx, sample_i in enumerate(sample_indices):
    fname, pred_lbl, conf, img_data = results[sample_i]
    plt.subplot(4, 4, idx + 1)
    plt.imshow(img_data.astype("uint8"))
    color = "green" if pred_lbl == "biodegradable" else "blue"
    plt.title(f"{fname[:20]}\n{pred_lbl} ({conf:.1%})", color=color, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig(output_plot_path, dpi=150)
plt.close()
print(f"\nSaved prediction visualization grid to {output_plot_path}")
