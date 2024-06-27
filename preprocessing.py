import os
import cv2
import numpy as np
import pickle

# Fungsi untuk membaca gambar, mengubah ukuran, dan mengonversi ke grayscale
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Fungsi untuk melabeli data berdasarkan nama file dan folder
def label_image(image_path):
    # Gunakan os.path.sep untuk pemisah direktori yang benar
    parts = os.path.normpath(image_path).split(os.path.sep)
    person_id = parts[-3]  # 001, 002, ..., 100
    finger_side = parts[-2]  # left, right
    finger_type = parts[-1].split('_')[0]  # index, middle, ring
    return f"{person_id}_{finger_side}_{finger_type}"

# Direktori dataset
dataset_dir = "dataset"

# List untuk menyimpan gambar dan label
images = []
labels = []

# Membaca gambar dari direktori
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".bmp"):
            image_path = os.path.join(root, file)
            image = preprocess_image(image_path)
            label = label_image(image_path)
            images.append(image)
            labels.append(label)

# Konversi list menjadi array
images = np.array(images)
labels = np.array(labels)

# Simpan hasil ke dalam file pickle
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump((images, labels), f)

print("Preprocessing selesai.")
