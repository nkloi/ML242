import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", category=UserWarning)

model = load_model("./model/model.keras")
image_path = "./images/class1.jpg"
img_list=[]

def image_capture(cam_id = 0, zoom = 0, number=0, save_dir="./" ,filename="image.png"):
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()
    print("Camera open successfully. Type 'space' for capture")
    # Example: Width = 640, Height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Failed to capture frame.")
            break

        zoom_factor = zoom

        height, width, _ = frame.shape
        new_height = int(height / zoom_factor)
        new_width = int(width / zoom_factor)
        x_offset = int((width - new_width) / 2)
        y_offset = int((height - new_height) / 2)

        zoomed_frame = frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

        frame = cv2.resize(zoomed_frame, (width, height))
        cv2.imshow('Camera for number ' + str(number), frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):  # Capture and save when 'space' is pressed
            file_path = os.path.join(save_dir, f"{filename}")

            # Save the frame as an image
            cv2.imwrite(file_path, frame)
            print(f"Image saved at: {file_path}")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def preprocess_image_one_digit(image_path):
    if not os.path.exists(image_path):
        print("Lỗi: Ảnh '"+image_path+"' không tồn tại!")
        return 0

    # Đọc ảnh gốc (Grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Lỗi: Không thể đọc {image_path}")
        return

    img = cv2.GaussianBlur(img, (15,15), 0)
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    # Apply OTSU thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    X = []
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            X.append(img[i][j])

    return X


def preprocess_image_multi_digits(image_path, output_folder = "./digits_split_output", padding = 10):
    if not os.path.exists(image_path):
        print("Lỗi: Ảnh '" + image_path + "' không tồn tại!")
        return 0

    # Đọc ảnh gốc (Grayscale)
    img_org = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_list.append(img_org)    #1

    if img_org is None:
        print(f"Lỗi: Không thể đọc {image_path}")
        return

    img = cv2.GaussianBlur(img_org, (5,5), 0)

    # Apply OTSU thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Giãn nở
    kernel = np.ones((5,5), np.uint8)  # Kích thước kernel lớn hơn để nối nét
    img = cv2.dilate(img, kernel, iterations=1)

    img_list.append(img)    #2

    # Tìm contours của các chữ số
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Chuyển ảnh về BGR để vẽ màu
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Vẽ contours lên ảnh
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    img_contours_org = img_contours.copy()

    # Sắp xếp contours theo tọa độ x để đúng thứ tự từ trái qua phải
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Split digits
    digit_regions = []
    for idx, contour in enumerate(contours, start = 1):
        x, y, w, h = cv2.boundingRect(contour)

        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, img.shape[1] - x)
        h = min(h + 2 * padding, img.shape[0] - y)

        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text_position = (x, y - 10)
        cv2.putText(img_contours, f"#{idx}", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2, cv2.LINE_AA)

        digit_regions.append((x, y, w, h))

    img_list.append(img_contours)  # 3

    # Xử lý dấu "=" - Merge box
    fixed_digit_regions = []
    for items in range(0, len(digit_regions) - 1):
        x0, y0, w0, h0 = digit_regions[items]
        x1, y1, w1, h1 = digit_regions[items + 1]

        disparity = abs(x1 - x0)
        if disparity <= 10:
            h0 = abs(y1 - y0) + h1
            w0 = max(w0, w1)
            x0 = min(x0, x1)
            y0 = min(y0, y1)
            items += 1

        fixed_digit_regions.append((x0, y0, w0, h0))
        if items == len(digit_regions) - 2 and disparity > 10:
            fixed_digit_regions.append((x1, y1, w1, h1))

    # Draw
    for idx in range(0, len(fixed_digit_regions)):
        x, y, w, h = fixed_digit_regions[idx]

        cv2.rectangle(img_contours_org, (x, y), (x + w, y + h), (0, 0, 255), 2)

        text_position = (x, y - 10)
        cv2.putText(img_contours_org, "#{}".format(idx + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2, cv2.LINE_AA)

    img_list.append(img_contours_org)   #4

    # Thư mục lưu ảnh
    if not os.path.exists("digit_split_raw_img"):
        os.makedirs("digit_split_raw_img")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cắt và lưu từng chữ số
    for i, (x, y, w, h) in enumerate(fixed_digit_regions):
        digit = img_org[y:y + h, x:x + w]
        cv2.imwrite(f"./digit_split_raw_img/digit_{i}.png", digit)

        digit = img[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        digit = abs(255 - digit)
        # digit = abs(digit - 255)

        cv2.imwrite(f"{output_folder}/digit_{i}.png", digit)

    print(f"Đã tách {len(fixed_digit_regions)} chữ số và lưu vào thư mục '{output_folder}'.")

    return len(fixed_digit_regions)


def all_processes_gui(img_list):
    rows, cols = 2, 2  # Ví dụ: 2 hàng, 2 cột
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(img_list): 
            ax.imshow(img_list[i], cmap='gray')
            ax.set_title(f"Image {i + 1}")
            ax.axis("off")  
    plt.tight_layout()
    plt.show()


def calculation(predictions):
    # Dictionary mapping prediction numbers to symbols
    symbol_map = {
        10: '/',
        11: '=',
        12: '-',
        13: '*',
        14: '+' 
    }
    
     # Chuyển đổi dự đoán thành chuỗi biểu thức
    expression_parts = []
    current_number = ""
    
    for pred in predictions:
        if pred in symbol_map:
            # Nếu gặp toán tử, lưu số hiện tại (nếu có) và thêm toán tử
            if current_number:
                expression_parts.append(current_number)
                current_number = ""
            expression_parts.append(symbol_map[pred])
        else:
            # Nếu là số, thêm vào số hiện tại
            current_number += str(pred)
    
    # Thêm số cuối cùng nếu có
    if current_number:
        expression_parts.append(current_number)
    
    # Tạo biểu thức toán học
    expression_str = " ".join(expression_parts)
    
    # Tìm vị trí dấu bằng
    if "=" in expression_parts:
        equal_index = expression_parts.index("=")
        left_expr = "".join(expression_parts[:equal_index]).replace(" ", "")
        right_expr = "".join(expression_parts[equal_index+1:]).replace(" ", "")
        
        # Tính toán vế trái
        try:
            left_result = eval(left_expr)
            
            # Nếu vế phải trống, thêm kết quả
            if not right_expr:
                expression_str += f" {left_result}"
            # Kiểm tra xem biểu thức đúng không
            elif right_expr:
                right_result = eval(right_expr)
                if left_result == right_result:
                    expression_str += f" (Đúng: {left_result} = {right_result})"
                else:
                    expression_str += f" (Sai: {left_result} ≠ {right_result})"
        except Exception as e:
            expression_str += f" (Lỗi: {str(e)})"
    # Nếu không có dấu bằng, thử tính toán biểu thức
    try:
        expr_to_eval = "".join(expression_parts).replace(" ", "")
        result = eval(expr_to_eval)
        expression_str += f" = {result}"
    except Exception as e:
        expression_str += f" (Lỗi: {str(e)})"
    
    return expression_str

if __name__ == "__main__":
    # image_capture(cam_id= 0, zoom = 2, number = 0, filename= image_path)
    img = preprocess_image_one_digit(image_path)
    numbers = preprocess_image_multi_digits(image_path, output_folder = "./digits_split_output", padding = 10)

    predictions = []
    for nums in range (0, numbers):
        img = cv2.imread("./digits_split_output/digit_"+str(nums)+".png", cv2.IMREAD_GRAYSCALE)

        # X for prediction
        X = img.reshape(-1,28,28,1)

        prediction = model.predict(X)
        print("Predicted digit {}:".format(nums + 1), np.argmax(prediction))

        predictions.append(np.argmax(prediction))

    print("Predictions:", ', '.join(map(str, predictions)))
    print("Expression:", calculation(predictions))
    all_processes_gui(img_list)

    cv2.destroyAllWindows()





