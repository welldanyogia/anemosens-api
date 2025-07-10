from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Load Keras model (arsitektur + bobot tersimpan di model_anemia.h5)
model = tf.keras.models.load_model('model_anemia.h5')
# Print arsitektur (opsional, untuk debugging):
# model.summary()

# Tentukan input size dari model
input_shape = model.input_shape[1:3]  # e.g., (224, 224)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Pastikan ada file image di form-data dengan key 'image'
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    # Buka dan ubah ukuran
    img = Image.open(file.stream).convert('RGB')
    img = img.resize(input_shape)
    x = np.array(img, dtype=np.float32)
    # Preprocessing sesuai ImageNet atau custom pipeline
    x = preprocess_input(x)  # jika model dilatih dengan preprocess_input
    x = np.expand_dims(x, axis=0)

    # Prediksi
    preds = model.predict(x)
    # Asumsikan model output [prob_tidak_anemia, prob_anemia]
    anemia_prob = float(preds[0][1])
    return jsonify({'anemia_probability': anemia_prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)