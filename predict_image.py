import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def predict(image_path, model_path="waste_classifier.keras"):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Build model architecture (alpha=0.5) to avoid version deserialization issues
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

    # Load and preprocess image
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        resized_img = rgb_img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(resized_img, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    
    class_names = ['biodegradable', 'non_biodegradable']
    pred_idx = 1 if prob >= 0.5 else 0
    pred_label = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else (1.0 - prob)

    print("\n==========================================")
    print(f" Image File  : {os.path.basename(image_path)}")
    print(f" Prediction  : {pred_label.upper()}")
    print(f" Confidence  : {confidence * 100:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_p = sys.argv[1]
    else:
        img_p = input("Enter path to waste image: ").strip('"')
    
    predict(img_p)
