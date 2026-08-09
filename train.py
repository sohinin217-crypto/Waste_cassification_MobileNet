import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
dataset_dir = r"d:\Dekstop\archive (3)\dataset"
model_save_path = r"d:\Dekstop\archive (3)\waste_classifier.keras"

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
EPOCHS = 15
SEED = 123

print("Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

print("\nLoading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

class_names = train_ds.class_names
print(f"\nDetected classes: {class_names}")
# Keras alphabetical ordering:
# class_names[0] = "biodegradable" (labels=0.0)
# class_names[1] = "non_biodegradable" (labels=1.0)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Data Augmentation (applied only during training)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# MobileNetV2 Backbone
# Use alpha=0.35, ImageNet weights, and exclude classification head (top)
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    alpha=0.5,
    include_top=False,
    weights='imagenet'
)
# Freeze the backbone
backbone.trainable = False

# Build the model
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = data_augmentation(inputs)
# Rescale pixels to [-1, 1] range as MobileNetV2 expects
x = layers.Rescaling(scale=1.0/127.5, offset=-1.0)(x)
x = backbone(x, training=False)  # Ensure batch normalization stays in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs, name="waste_classifier")

model.summary()

# Compile
print("\nCompiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the Keras model
print(f"\nSaving model to {model_save_path}...")
model.save(model_save_path)

# Plot training & validation curves
print("\nPlotting training curves...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(r"d:\Dekstop\archive (3)\training_curves.png", dpi=150)
plt.close()
print("Saved training_curves.png")

# Run full evaluation to calculate confusion matrix and metrics
print("\nEvaluating model on validation dataset...")
all_true_labels = []
all_preds = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    all_true_labels.extend(labels.numpy().flatten())
    all_preds.extend(preds.flatten())

all_true_labels = np.array(all_true_labels)
all_preds = np.array(all_preds)
all_pred_labels = (all_preds >= 0.5).astype(int)

# 2x2 Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels)
print("\nConfusion Matrix:")
print("---------------------------------------------")
print(f"True\\Pred | Biodegradable | Non-Biodegradable")
print("---------------------------------------------")
print(f"Bio       | {cm[0][0]:^13} | {cm[0][1]:^17}")
print(f"Non-Bio   | {cm[1][0]:^13} | {cm[1][1]:^17}")
print("---------------------------------------------")

print("\nClassification Report:")
print(classification_report(all_true_labels, all_pred_labels, target_names=class_names))

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(r"d:\Dekstop\archive (3)\confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# Generate and plot predictions for visual check
print("\nGenerating sample predictions visualization...")
for images, labels in val_ds.take(1):
    preds = model.predict(images, verbose=0)
    
    # Text printout
    print("\nSample Predictions:")
    print(f"{'Index':<6} | {'True Class':<18} | {'Predicted Class':<18} | {'Confidence':<10} | {'Status':<10}")
    print("-" * 72)
    
    plt.figure(figsize=(12, 12))
    for i in range(min(16, len(images))):
        true_lbl = int(labels[i][0])
        prob = float(preds[i][0])
        pred_lbl = 1 if prob >= 0.5 else 0
        conf = prob if pred_lbl == 1 else (1.0 - prob)
        
        true_name = class_names[true_lbl]
        pred_name = class_names[pred_lbl]
        status = "CORRECT" if true_lbl == pred_lbl else "INCORRECT"
        
        print(f"{i:<6} | {true_name:<18} | {pred_name:<18} | {conf:.4f} | {status:<10}")
        
        # Grid plotting
        plt.subplot(4, 4, i + 1)
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)
        title_color = 'green' if true_lbl == pred_lbl else 'red'
        plt.title(f"True: {true_name}\nPred: {pred_name} ({conf:.2%})", color=title_color, fontsize=8)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(r"d:\Dekstop\archive (3)\sample_predictions.png", dpi=150)
    plt.close()
    print("Saved sample_predictions.png")

print("\nTraining and evaluation pipeline execution completed successfully!")
