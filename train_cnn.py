import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Folder dataset
DATASET_DIR = 'dataset'
CATEGORIES = ['mentah', 'matang']
IMG_SIZE = 128

def load_images():
    data, labels = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(label)
    return np.array(data), np.array(labels)

X, y = load_images()
X = X / 255.0  # Normalisasi
y = to_categorical(y, 2)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)

# Simpan model
model.save('cnn_model.h5')
print("Model CNN berhasil disimpan sebagai cnn_model.h5")
