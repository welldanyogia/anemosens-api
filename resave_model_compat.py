#!/usr/bin/env python3
"""
Resave the model in a portable format compatible with older TensorFlow versions.
Run this on your local machine before deploying.
"""
import tensorflow as tf
import os

MODEL_PATH = "model_anemia_v2.h5"
OUTPUT_PATH = "model_anemia_v2.keras"

print(f"TensorFlow version: {tf.__version__}")
print(f"Loading model from: {MODEL_PATH}")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# Re-save in .keras format (native TF 2.x format)
print(f"\nSaving to: {OUTPUT_PATH}")
model.save(OUTPUT_PATH)
print("✅ Model saved successfully in .keras format!")
