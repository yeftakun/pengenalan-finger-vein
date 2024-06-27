# training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Memuat data dari file pickle
with open("preprocessed_data.pkl", "rb") as f:
    images, labels = pickle.load(f)

# Normalisasi gambar
images = images / 255.0

# Konversi label menjadi one-hot encoding
unique_labels = np.unique(labels)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_map[label] for label in labels])
labels = to_categorical(labels, num_classes=len(unique_labels))

# Membagi data menjadi training dan validation set
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Membuat model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Menyimpan model
model.save("finger_vein_model.h5")

# Simpan label_map ke dalam file pickle
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training selesai.")
