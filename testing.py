import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Memuat model yang telah dilatih
model = load_model("finger_vein_model.h5")

# Fungsi untuk preprocess gambar uji
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    return image.reshape(1, 128, 128, 1)

# Memuat label_map dari file pickle
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Membalik label_map untuk mendapatkan label dari indeks
reverse_label_map = {idx: label for label, idx in label_map.items()}

# Contoh penggunaan pada satu gambar uji
test_image_path = "input/vein-input.bmp"  # Ganti dengan path gambar uji yang sesuai
test_image = preprocess_image(test_image_path)

# Melakukan prediksi
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction)
confidence_score = np.max(prediction)  # Mendapatkan persentase kecocokan

print(f"Hasil prediksi: {reverse_label_map[predicted_label]}")
print(f"Persentase kecocokan: {confidence_score * 100:.2f}%")

print("Testing selesai.")
