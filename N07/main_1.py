import cv2
import numpy as np
import tensorflow as tf
import time
from mediapipe.python.solutions import hands as mp_hands

# 1. Load model đã train
model = tf.keras.models.load_model('sign_language_model.keras')

# 2. Danh sách labels từ A-Z
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# 3. Khởi tạo MediaPipe Hands
mp_hands_module = mp_hands
hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 4. Hàm trích xuất keypoints + bounding box
def detect_hand(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            h, w, _ = frame.shape
            min_x, min_y = w, h
            max_x, max_y = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
                keypoints.extend([lm.x, lm.y, lm.z])

            return np.array(keypoints), (min_x, min_y, max_x, max_y)
    else:
        return None, None

# 5. Mở webcam
cap = cv2.VideoCapture(0)

detected_text = ""              # Chuỗi ký tự đã nhận diện
last_hand_time = time.time()   # Thời gian cuối cùng phát hiện tay
last_save_time = 0             # Thời gian cuối cùng lưu ký tự
save_interval = 3              # Thời gian chờ giữa các lần lưu ký tự (giây)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Lật gương
    current_time = time.time()

    keypoints, bbox = detect_hand(frame)

    if keypoints is not None:
        last_hand_time = current_time  # ✅ Cập nhật khi có tay

        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints, verbose=0)
        predicted_label = labels[np.argmax(prediction)]

        # ✅ Chỉ lưu ký tự nếu đã qua 3 giây kể từ lần lưu trước
        if current_time - last_save_time >= save_interval:
            detected_text += predicted_label
            last_save_time = current_time

        # Vẽ bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0, 255, 0), 3)

        # Hiển thị nhãn dự đoán
        cv2.putText(frame, f'{predicted_label}', (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        cv2.putText(frame, "No hand detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị chuỗi đã nhận diện
    cv2.putText(frame, f'Text: {detected_text}', (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị khung hình
    cv2.imshow('Sign Language Detection', frame)

    # ✅ Thoát nếu không phát hiện tay trong 5 giây
    if current_time - last_hand_time > 5:
        print(f'\n=== Chuỗi đã nhận diện: {detected_text} ===')
        break

    # Thoát thủ công nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f'\n=== Chuỗi đã nhận diện: {detected_text} ===')
        break

# Đóng camera và cửa sổ
cap.release()
cv2.destroyAllWindows()
