import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# --- CẤU HÌNH ---
IMG_HEIGHT, IMG_WIDTH = 96, 128
MODEL_PATH = "btl/cnn.h5"
ENCODER_PATH = "btl/cnn-label_encoder_classes.npy"
TEST_CSV = r"btl\processed_split_data\test_labels.csv"
TEST_IMG_FOLDER = r"btl\processed_split_data\test"

# --- TẢI DỮ LIỆU TEST ---
df_test = pd.read_csv(TEST_CSV)
images, labels = [], []

for _, row in df_test.iterrows():
    img_path = os.path.join(TEST_IMG_FOLDER, row['image'])
    img = load_img(img_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    images.append(img_array)
    labels.append(row['label'])

X_test = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y_test_raw = np.array(labels)

# --- MÃ HÓA NHÃN ---
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(ENCODER_PATH, allow_pickle=True)
y_test_true = label_encoder.transform(y_test_raw)

# --- TẢI MÔ HÌNH ---
model = load_model(MODEL_PATH)

# --- DỰ ĐOÁN ---
y_pred_probs = model.predict(X_test, verbose=1)
y_test_pred = np.argmax(y_pred_probs, axis=1)

# --- ĐÁNH GIÁ ---
acc = accuracy_score(y_test_true, y_test_pred)
print(f"\n✅ Độ chính xác trên tập test: {acc:.4f}\n")
print("📊 Classification report:")
print(classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_))

# --- HIỂN THỊ ẢNH DỰ ĐOÁN SAI ---
wrong_idx = np.where(y_test_true != y_test_pred)[0]
print(f"Số ảnh đoán sai: {len(wrong_idx)}")

plt.figure(figsize=(12, 6))
for i, idx in enumerate(wrong_idx[:8]):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_test[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
    true_label = label_encoder.inverse_transform([y_test_true[idx]])[0]
    pred_label = label_encoder.inverse_transform([y_test_pred[idx]])[0]
    plt.title(f"T: {true_label} / P: {pred_label}", fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- TÍNH CONFUSION MATRIX ---
cm = confusion_matrix(y_test_true, y_test_pred)
class_names = label_encoder.classes_

# --- TÍNH ĐỘ CHÍNH XÁC TỪNG LỚP ---
class_acc = cm.diagonal() / cm.sum(axis=1)  # Accuracy per class

# --- TÌM 10 LỚP NHẬN DIỆN TỐT NHẤT VÀ KÉM NHẤT ---
best_idx = np.argsort(class_acc)[-10:][::-1]  # Top 10 tốt nhất
worst_idx = np.argsort(class_acc)[:10]        # Top 10 kém nhất

# --- VẼ CONFUSION MATRIX CHO 10 LỚP TỐT NHẤT ---
best_labels = [class_names[i] for i in best_idx]
cm_best = cm[np.ix_(best_idx, best_idx)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens',
            xticklabels=best_labels, yticklabels=best_labels)
plt.title("Confusion Matrix - Top 10 Class Accuracy")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- VẼ CONFUSION MATRIX CHO 10 LỚP KÉM NHẤT ---
worst_labels = [class_names[i] for i in worst_idx]
cm_worst = cm[np.ix_(worst_idx, worst_idx)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_worst, annot=True, fmt='d', cmap='Oranges',
            xticklabels=worst_labels, yticklabels=worst_labels)
plt.title("Confusion Matrix - 10 Worst Recognized Classes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
from sklearn.metrics import classification_report
import pandas as pd

# --- TẠO BẢNG TỪ classification_report ---
report_dict = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

# Chuyển thành DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Hiển thị nhanh bảng
print("\n📊 Classification Report - Bảng:")
print(report_df.head())  # hoặc print(report_df.to_string()) để xem đầy đủ

# --- LƯU RA FILE CSV ---
report_df.to_csv("classification_report.csv", index=True)
print("\n📁 Đã lưu bảng báo cáo vào file classification_report.csv")
