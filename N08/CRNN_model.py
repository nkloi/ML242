import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Cấu hình
IMG_HEIGHT, IMG_WIDTH = 96, 128
IMAGE_FOLDER = r"btl\processed_split_data\train"
CSV_PATH = r"btl\processed_split_data\train_labels.csv"

# Load dữ liệu
df = pd.read_csv(CSV_PATH)
images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['image'])
    img = load_img(img_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    images.append(img_array)
    labels.append(row['label'])

X = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y_raw = np.array(labels)
print(y_raw)

# Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
NUM_CLASSES = len(label_encoder.classes_)
print(NUM_CLASSES)
# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình CNN cải tiến
model = models.Sequential([
  layers.Input(shape=(96, 128, 1)),
  layers.Conv2D(32, (3, 3), padding='same'),
  layers.BatchNormalization(),  # Thêm batch normalization
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Dropout(0.25),
  
  layers.Conv2D(64, (3, 3), padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Dropout(0.35),
  
  layers.Conv2D(128, (3, 3), padding='same'),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Dropout(0.35),
  
  layers.Reshape((16, 12 * 128)),
  
  layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
  layers.BatchNormalization(),
  layers.Bidirectional(layers.LSTM(64)),
  layers.BatchNormalization(),
  
  layers.Dense(64),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.Dropout(0.5),
  layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=4)
]

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

# Lưu model và label encoder
model.save("cnn.h5")
np.save("cnn-label_encoder_classes.npy", label_encoder.classes_)
print("\nĐã lưu mô hình và encoder.")

# Vẽ biểu đồ Loss và Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
