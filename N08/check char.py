import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# --- CẤU HÌNH ---
IMG_HEIGHT, IMG_WIDTH = 96, 128
MODEL_PATH = "btl/cnn.h5"
ENCODER_PATH = "btl/cnn-label_encoder_classes.npy"
IMAGE_PATH = r"btl\check\gcap.png"  # <<< Đặt đường dẫn ảnh tại đây

# --- Resize giữ tỉ lệ + Padding ---
def resize_and_pad(img, target_size):
    old_size = img.size  # (width, height)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    new_img = Image.new("L", target_size, 255)
    paste_position = ((target_size[0] - new_size[0]) // 2,
                      (target_size[1] - new_size[1]) // 2)
    new_img.paste(img, paste_position)
    return new_img

# --- TIỀN XỬ LÝ ẢNH: Tách nền bằng Otsu + Crop + Resize ---
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)

    # Otsu thresholding (chữ đen, nền trắng)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary == 0  # True tại vùng chữ (đen)

    # Bounding box ký tự
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        raise ValueError("Không tìm thấy ký tự trong ảnh.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_np = binary[y_min:y_max + 1, x_min:x_max + 1]
    cropped_img = Image.fromarray(cropped_np)

    # Resize giữ tỉ lệ và pad về đúng kích thước
    padded_img = resize_and_pad(cropped_img, (IMG_WIDTH, IMG_HEIGHT))

    # Chuẩn hóa
    img_array = np.array(padded_img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array, padded_img

# --- DỰ ĐOÁN ---
def predict_character(image_path):
    model = load_model(MODEL_PATH)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(ENCODER_PATH, allow_pickle=True)

    img_array, processed_img = preprocess_image(image_path)

    probs = model.predict(img_array)[0]
    top5_idx = np.argsort(probs)[::-1][:5]

    # Hiển thị ảnh sau xử lý kèm dự đoán
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Dự đoán: {label_encoder.classes_[top5_idx[0]]} ({probs[top5_idx[0]]:.2f})")
    plt.axis('off')
    plt.show()

    print("\n🔍 Top 5 dự đoán:")
    for idx in top5_idx:
        print(f"{label_encoder.classes_[idx]}: {probs[idx]:.4f}")

    confidence = probs[top5_idx[0]]
    if confidence < 0.6:
        print("⚠️ Cảnh báo: Dự đoán không chắc chắn (confidence < 60%)")

    return label_encoder.classes_[top5_idx[0]]

# --- MAIN ---
if __name__ == "__main__":
    result = predict_character(IMAGE_PATH)
    print(f"\n✅ Ký tự được nhận diện: {result}")
