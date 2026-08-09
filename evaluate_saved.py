import os
import tensorflow as tf
import numpy as np

# Paths
dataset_dir = r"d:\Dekstop\archive (3)\dataset"
model_path = r"d:\Dekstop\archive (3)\waste_classifier.keras"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

print("Loading dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nEvaluating model on validation dataset...")
loss, accuracy = model.evaluate(val_ds)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")

# Show a few sample predictions
print("\nSample Predictions:")
class_names = val_ds.class_names
for images, labels in val_ds.take(1):
    preds = model.predict(images, verbose=0)
    print(f"{'Index':<6} | {'True Class':<18} | {'Predicted Class':<18} | {'Confidence':<10}")
    print("-" * 62)
    for i in range(min(5, len(images))):
        true_lbl = int(labels[i][0])
        prob = float(preds[i][0])
        pred_lbl = 1 if prob >= 0.5 else 0
        conf = prob if pred_lbl == 1 else (1.0 - prob)
        
        true_name = class_names[true_lbl]
        pred_name = class_names[pred_lbl]
        print(f"{i:<6} | {true_name:<18} | {pred_name:<18} | {conf:.2%}")
