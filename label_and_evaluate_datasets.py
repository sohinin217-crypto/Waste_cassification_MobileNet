import os
import glob
import shutil
import numpy as np
import tensorflow as tf

datasets_dir = r"d:\Dekstop\archive (3)\Datasets"
model_path = r"d:\Dekstop\archive (3)\waste_classifier.keras"

# Move/clean up old unsegregated subfolder if present so only binary folders exist
unseg_dir = os.path.join(datasets_dir, "domestic-trash-garbage-dataset")
if os.path.exists(unseg_dir):
    shutil.rmtree(unseg_dir, ignore_errors=True)

bio_dir = os.path.join(datasets_dir, "biodegradable")
non_bio_dir = os.path.join(datasets_dir, "non_biodegradable")

print(f"Evaluating model on ground-truth categorized Datasets at: {datasets_dir}")
print(f"  Biodegradable folder count    : {len(os.listdir(bio_dir))} images")
print(f"  Non-Biodegradable folder count: {len(os.listdir(non_bio_dir))} images")

# Build MobileNetV2 (alpha=0.5) model architecture
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
model.load_weights(model_path)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

eval_ds = tf.keras.utils.image_dataset_from_directory(
    datasets_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

loss, accuracy = model.evaluate(eval_ds)

print("\n==================================================")
print(f" TOTAL VALIDATION ACCURACY ON DATASETS: {accuracy * 100:.2f}%")
print(f" TOTAL VALIDATION LOSS ON DATASETS    : {loss:.4f}")
print("==================================================\n")
