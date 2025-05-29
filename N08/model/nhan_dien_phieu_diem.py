import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pytesseract
import re
import os

class VietnameseAnswerSheetRecognizer:
    def __init__(self, 
                 character_model_path="btl/cnn.h5", 
                 encoder_path="btl/cnn-label_encoder_classes.npy",
                 img_height=96, 
                 img_width=128,
                 force_cnn=True):  # Added force_cnn parameter
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.MODEL_PATH = character_model_path
        self.ENCODER_PATH = encoder_path
        self.force_cnn = force_cnn  # Whether to force using CNN over OCR
        
        # Tải mô hình nhận dạng ký tự nếu đường dẫn hợp lệ
        if os.path.exists(character_model_path) and os.path.exists(encoder_path):
            self.char_model = load_model(character_model_path)
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)
            print("✅ Đã tải mô hình nhận dạng ký tự thành công!")
            self.use_cnn = True
        else:
            self.char_model = None
            self.label_encoder = None
            self.use_cnn = False
            print("⚠️ Không tìm thấy mô hình nhận dạng ký tự, sẽ sử dụng Tesseract OCR!")
        
        # Kiểm tra xem Tesseract đã được cài đặt chưa
        try:
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            print("✅ Đã phát hiện Tesseract OCR!")
        except:
            self.use_tesseract = False
            print("⚠️ Không phát hiện Tesseract OCR. Vui lòng cài đặt Tesseract để sử dụng tính năng OCR.")
    
    def process_image(self, image_path):
        """Xử lý ảnh từ đường dẫn hoặc mảng numpy"""
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ file: {image_path}")
        else:
            image = image_path
            
        # Lưu lại ảnh gốc để hiển thị
        original = image.copy()
            
        # Chuyển sang ảnh gray nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Làm nét ảnh
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        return binary, original

    def detect_form_areas(self, binary_image):
        """Phát hiện các vùng chính trong phiếu: thông tin cá nhân và vùng bubbles"""
        # Tìm đường viền trong ảnh
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc các contour lớn, có thể là các vùng chứa thông tin
        form_areas = []
        height, width = binary_image.shape
        min_area = width * height * 0.01  # Diện tích tối thiểu là 1% diện tích ảnh
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                form_areas.append((x, y, w, h))
        
        # Sắp xếp các vùng theo tọa độ y (từ trên xuống dưới)
        form_areas = sorted(form_areas, key=lambda area: area[1])
        
        return form_areas
    
    def extract_text_with_blue_color(self, original_image):
        """Trích xuất văn bản màu xanh dương từ ảnh và chuẩn bị cho nhận dạng CNN"""
        # Chuyển từ BGR sang HSV để dễ dàng phát hiện màu
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa phạm vi màu xanh dương trong HSV
        # Có thể điều chỉnh ngưỡng này để phù hợp với sắc thái xanh cụ thể
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Tạo mặt nạ cho màu xanh
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Áp dụng mặt nạ để lấy chỉ phần màu xanh
        blue_text = cv2.bitwise_and(original_image, original_image, mask=blue_mask)
        
        # Chuyển đổi sang ảnh xám
        gray_blue_text = cv2.cvtColor(blue_text, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng ngưỡng để có được văn bản đen trên nền trắng
        # Sử dụng THRESH_OTSU để tự động xác định ngưỡng tối ưu
        _, binary_text = cv2.threshold(gray_blue_text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Đảo ngược màu sắc để có chữ đen trên nền trắng (nếu cần)
        binary_text = cv2.bitwise_not(binary_text)
        
        # Áp dụng một số xử lý hình thái học để cải thiện chất lượng văn bản
        kernel = np.ones((2, 2), np.uint8)
        binary_text = cv2.morphologyEx(binary_text, cv2.MORPH_CLOSE, kernel)
        
        # Hiển thị các bước xử lý để debug
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.title("Ảnh gốc")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("Mặt nạ màu xanh")
        plt.imshow(blue_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("Văn bản màu xanh được trích xuất")
        plt.imshow(cv2.cvtColor(blue_text, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("Ảnh xám")
        plt.imshow(gray_blue_text, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("Văn bản nhị phân")
        plt.imshow(binary_text, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return binary_text, blue_mask
    
    def recognize_text_with_cnn(self, binary_image):
        """Nhận dạng văn bản sử dụng mô hình CNN đã tải"""
        if self.char_model is None:
            print("⚠️ Không có mô hình CNN được tải, chuyển sang sử dụng Tesseract")
            return self.perform_ocr(binary_image)
        
        # Tìm các contour văn bản
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc các contour quá nhỏ
        min_contour_area = 30  # Điều chỉnh theo kích thước chữ của bạn
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # Sắp xếp các contour từ trái sang phải, trên xuống dưới
        # Đầu tiên nhóm các contour theo dòng
        y_sorted = sorted(valid_contours, key=lambda cnt: cv2.boundingRect(cnt)[1])
        
        if not y_sorted:
            return ""
            
        # Tìm chiều cao trung bình của một dòng
        avg_height = np.mean([cv2.boundingRect(cnt)[3] for cnt in y_sorted])
        line_threshold = avg_height * 0.7  # Ngưỡng để xác định các ký tự cùng dòng
        
        # Nhóm các contour theo dòng
        current_line = [y_sorted[0]]
        lines = []
        current_y = cv2.boundingRect(y_sorted[0])[1]
        
        for cnt in y_sorted[1:]:
            y = cv2.boundingRect(cnt)[1]
            if abs(y - current_y) <= line_threshold:
                current_line.append(cnt)
            else:
                # Sắp xếp các ký tự trong dòng từ trái sang phải
                lines.append(sorted(current_line, key=lambda c: cv2.boundingRect(c)[0]))
                current_line = [cnt]
                current_y = y
        
        # Thêm dòng cuối cùng
        if current_line:
            lines.append(sorted(current_line, key=lambda c: cv2.boundingRect(c)[0]))
        
        # Hiển thị các contour được phát hiện
        contour_image = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            for cnt in line:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(contour_image, (x, y), (x+w, y+h), color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.title("Các ký tự được phát hiện")
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        # Nhận dạng từng ký tự bằng mô hình CNN
        recognized_text = ""
        for line in lines:
            line_text = ""
            for cnt in line:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Cắt ký tự và thêm padding
                # Added safety checks for index boundaries
                y_start = max(0, y-2)
                y_end = min(binary_image.shape[0], y+h+2)
                x_start = max(0, x-2)
                x_end = min(binary_image.shape[1], x+w+2)
                
                char_img = binary_image[y_start:y_end, x_start:x_end]
                if char_img.size == 0:
                    continue
                    
                # Đảm bảo có đủ padding
                char_img = cv2.copyMakeBorder(char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
                
                # Thay đổi kích thước cho phù hợp với mô hình
                resized_char = cv2.resize(char_img, (self.IMG_WIDTH, self.IMG_HEIGHT))
                
                # Chuẩn hóa
                normalized_char = resized_char / 255.0
                
                # Mở rộng chiều
                input_char = np.expand_dims(normalized_char, axis=-1)
                input_char = np.expand_dims(input_char, axis=0)
                
                # Dự đoán
                prediction = self.char_model.predict(input_char, verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_char = self.label_encoder.inverse_transform([predicted_class])[0]
                
                line_text += predicted_char
                
            recognized_text += line_text + "\n"
        
        return recognized_text.strip()

    def perform_ocr(self, image, lang='vie'):
        """Thực hiện OCR trên ảnh sử dụng Tesseract"""
        # Nếu force_cnn được bật và mô hình CNN khả dụng, không sử dụng OCR
        if self.force_cnn and self.use_cnn:
            print("⚠️ Force CNN được bật, không sử dụng OCR")
            return "Không sử dụng OCR vì force_cnn được bật"
            
        if not self.use_tesseract:
            return "OCR không khả dụng (Tesseract chưa được cài đặt)"
        
        # Tối ưu ảnh cho OCR
        # Đảo ngược ảnh nếu là ảnh nhị phân
        if len(image.shape) == 2:
            image = cv2.bitwise_not(image)
        
        # Thực hiện OCR
        try:
            text = pytesseract.image_to_string(image, lang=lang, config='--psm 6')
            return text.strip()
        except Exception as e:
            print(f"Lỗi khi thực hiện OCR: {e}")
            return ""
    
    def extract_field_from_text(self, text, field_name, regex_pattern=None):
        """Trích xuất trường cụ thể từ văn bản OCR"""
        if not text:
            return None
            
        lines = text.split('\n')
        for line in lines:
            # Tìm kiếm dựa trên tên trường
            if field_name.lower() in line.lower():
                # Nếu có mẫu regex, sử dụng regex để trích xuất giá trị
                if regex_pattern:
                    match = re.search(regex_pattern, line)
                    if match:
                        return match.group(1).strip()
                else:
                    # Nếu không có regex, lấy phần sau dấu ":" nếu có
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    else:
                        # Nếu không có dấu ":", lấy phần không chứa tên trường
                        value = line.lower().replace(field_name.lower(), '').strip()
                        if value:
                            return value
        
        return None
    
    def extract_fields_from_form(self, original_image, binary_image):
        """Trích xuất thông tin từ phiếu sử dụng CNN cho văn bản màu xanh"""
        # Phát hiện văn bản màu xanh và chuyển thành ảnh nhị phân
        blue_text_binary, blue_text_mask = self.extract_text_with_blue_color(original_image)
        
        # Hiển thị mặt nạ text màu xanh
        plt.figure(figsize=(10, 6))
        plt.title("Văn bản màu xanh được phát hiện và xử lý")
        plt.imshow(blue_text_binary, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Sử dụng mô hình CNN để nhận dạng văn bản từ ảnh nhị phân
        # Luôn sử dụng CNN nếu có thể
        if self.use_cnn:
            recognized_text = self.recognize_text_with_cnn(blue_text_binary)
            print(f"Văn bản được nhận dạng bằng CNN:\n{recognized_text}")
        else:
            recognized_text = self.perform_ocr(blue_text_binary)
            print(f"Văn bản được nhận dạng bằng OCR:\n{recognized_text}")
        
        # Tiếp tục xử lý các vùng form như bình thường
        form_areas = self.detect_form_areas(binary_image)
        form_info = {}
        
        # Xác định vùng thông tin kỳ thi, môn thi, ngày thi
        header_area = None
        for x, y, w, h in form_areas:
            if y < original_image.shape[0] * 0.2:  # Vùng header thường nằm ở 20% đầu của ảnh
                header_area = original_image[y:y+h, x:x+w]
                break
        
        if header_area is not None:
            # Phát hiện text màu xanh trong vùng header
            header_blue_binary, header_blue_mask = self.extract_text_with_blue_color(header_area)
            
            # Hiển thị vùng header và text màu xanh
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Vùng header")
            plt.imshow(cv2.cvtColor(header_area, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Text màu xanh trong header")
            plt.imshow(header_blue_mask, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Sử dụng CNN để nhận dạng văn bản nếu có thể
            if self.use_cnn:
                header_text = self.recognize_text_with_cnn(header_blue_binary)
                print(f"Header được nhận dạng bằng CNN:\n{header_text}")
            else:
                header_text = self.perform_ocr(header_area)
                print(f"Header được nhận dạng bằng OCR:\n{header_text}")
            
            # Trích xuất thông tin về môn thi và ngày thi
            mon_thi = self.extract_field_from_text(header_text, "môn thi", r"[Mm]ôn\s*(?:thi)?:?\s*(.+)")
            ngay_thi = self.extract_field_from_text(header_text, "ngày thi", r"[Nn]gày\s*(?:thi)?:?\s*(\d{1,2}/\d{1,2}/\d{4})")
            
            if mon_thi:
                form_info["mon_thi"] = mon_thi
            if ngay_thi:
                form_info["ngay_thi"] = ngay_thi
        
        # Xác định vùng thông tin cá nhân
        personal_info_area = None
        for x, y, w, h in form_areas:
            if y > original_image.shape[0] * 0.2 and y < original_image.shape[0] * 0.6:
                personal_info_area = original_image[y:y+h, x:x+w]
                break
        
        if personal_info_area is not None:
            # Phát hiện text màu xanh trong vùng thông tin cá nhân
            personal_blue_binary, personal_blue_mask = self.extract_text_with_blue_color(personal_info_area)
            
            # Hiển thị vùng thông tin cá nhân và text màu xanh
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Vùng thông tin cá nhân")
            plt.imshow(cv2.cvtColor(personal_info_area, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Text màu xanh trong thông tin cá nhân")
            plt.imshow(personal_blue_mask, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Sử dụng CNN để nhận dạng văn bản nếu có thể
            if self.use_cnn:
                personal_text = self.recognize_text_with_cnn(personal_blue_binary)
                print(f"Thông tin cá nhân được nhận dạng bằng CNN:\n{personal_text}")
            else:
                personal_text = self.perform_ocr(personal_info_area)
                print(f"Thông tin cá nhân được nhận dạng bằng OCR:\n{personal_text}")
            
            # Trích xuất các thông tin cá nhân
            hoi_dong_thi = self.extract_field_from_text(personal_text, "hội đồng thi", r"[Hh]ội\s*[Đđ]ồng\s*(?:thi)?:?\s*(.+)")
            diem_thi = self.extract_field_from_text(personal_text, "điểm thi", r"[Đđ]iểm\s*(?:thi)?:?\s*(.+)")
            phong_thi = self.extract_field_from_text(personal_text, "phòng thi", r"[Pp]hòng\s*(?:thi)?:?\s*(.+)")
            ho_ten = self.extract_field_from_text(personal_text, "họ và tên", r"[Hh]ọ\s*(?:và)?\s*[Tt]ên:?\s*(.+)")
            ngay_sinh = self.extract_field_from_text(personal_text, "ngày sinh", r"[Nn]gày\s*(?:sinh)?:?\s*(\d{1,2}/\d{1,2}/\d{4})")
            
            # Thêm thông tin vào kết quả
            if hoi_dong_thi:
                form_info["hoi_dong_thi"] = hoi_dong_thi
            if diem_thi:
                form_info["diem_thi"] = diem_thi
            if phong_thi:
                form_info["phong_thi"] = phong_thi
            if ho_ten:
                form_info["ho_ten"] = ho_ten
            if ngay_sinh:
                form_info["ngay_sinh"] = ngay_sinh
        
        # Xác định vùng số báo danh và mã đề thi
        id_area = None
        for x, y, w, h in form_areas:
            if x > original_image.shape[1] * 0.7:  # Vùng này thường nằm ở bên phải
                id_area = original_image[y:y+h, x:x+w]
                break
        
        if id_area is not None:
            # Phát hiện text màu xanh trong vùng ID
            id_blue_binary, id_blue_mask = self.extract_text_with_blue_color(id_area)
            
            # Hiển thị vùng số báo danh và mã đề thi
            plt.figure(figsize=(6, 8))
            plt.title("Vùng số báo danh và mã đề thi")
            plt.imshow(cv2.cvtColor(id_area, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Sử dụng CNN để nhận dạng văn bản nếu có thể
            if self.use_cnn:
                id_text = self.recognize_text_with_cnn(id_blue_binary)
                print(f"ID được nhận dạng bằng CNN:\n{id_text}")
            else:
                id_text = self.perform_ocr(id_area)
                print(f"ID được nhận dạng bằng OCR:\n{id_text}")
            
            # Trích xuất số báo danh và mã đề thi
            so_bao_danh = self.extract_field_from_text(id_text, "số báo danh", r"[Ss]ố\s*[Bb]áo\s*[Dd]anh:?\s*(\d+)")
            ma_de_thi = self.extract_field_from_text(id_text, "mã đề thi", r"[Mm]ã\s*[Đđ]ề(?:\s*[Tt]hi)?:?\s*(\d+)")
            
            # Nếu không tìm thấy bằng regex, thử phương pháp đơn giản hơn: tìm chuỗi số
            if not so_bao_danh:
                numbers = re.findall(r'\d+', id_text)
                if len(numbers) >= 1:
                    so_bao_danh = numbers[0]
            
            if not ma_de_thi:
                numbers = re.findall(r'\d+', id_text)
                if len(numbers) >= 2:
                    ma_de_thi = numbers[1]
            
            if so_bao_danh:
                form_info["so_bao_danh"] = so_bao_danh
            if ma_de_thi:
                form_info["ma_de_thi"] = ma_de_thi
        
        return form_info
    def detect_answer_bubbles(self, binary_image, original_image):
        """Phát hiện các ô bubble để tô đáp án"""
        # Vùng bubble thường nằm ở nửa dưới bên phải của phiếu
        height, width = binary_image.shape
        
        # Added safety checks to avoid index errors
        h_half = int(height*0.6)
        w_half = int(width*0.5)
        
        if h_half >= height or w_half >= width:
            print("⚠️ Kích thước ảnh quá nhỏ để xác định vùng bubble")
            # Use full image in case of small images
            bubble_region = binary_image.copy()
            original_bubble_region = original_image.copy()
        else:
            bubble_region = binary_image[h_half:, w_half:]
            original_bubble_region = original_image[h_half:, w_half:]
        
        # Hiển thị vùng bubble
        plt.figure(figsize=(8, 10))
        plt.title("Vùng chứa bubbles đáp án")
        plt.imshow(cv2.cvtColor(original_bubble_region, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Added error handling for Hough Circle Transform
        try:
            # Phát hiện các hình tròn trong vùng bubble bằng Hough Circle Transform
            # Ensure bubble_region is not empty
            if bubble_region.size == 0:
                print("⚠️ Vùng bubble rỗng, không thể tìm các hình tròn")
                return []
                
            # Blur and invert to enhance circle detection
            blurred = cv2.GaussianBlur(bubble_region, (5, 5), 0)
            
            # Ensure blurred image is not empty
            if blurred.size == 0:
                print("⚠️ Lỗi khi làm mờ ảnh")
                return []
                
            # Phát hiện các hình tròn
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,  # Khoảng cách tối thiểu giữa các hình tròn
                param1=50,   # Ngưỡng cạnh cao trong phát hiện Canny
                param2=25,   # Ngưỡng phát hiện hình tròn
                minRadius=8,  # Bán kính tối thiểu của hình tròn
                maxRadius=15  # Bán kính tối đa của hình tròn
            )
            
            # Nếu không tìm thấy hình tròn nào
            if circles is None:
                print("⚠️ Không tìm thấy hình tròn nào trong vùng bubble")
                return []
            
            # Chuyển đổi kết quả sang định dạng phù hợp
            circles = np.uint16(np.around(circles))
            
            # Thêm offset để đưa về tọa độ trong ảnh gốc
            for circle in circles[0, :]:
                circle[0] += w_half
                circle[1] += h_half
            
            # Hiển thị kết quả phát hiện hình tròn
            result_image = original_image.copy()
            for i in circles[0, :]:
                # Vẽ các hình tròn được phát hiện
                cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Vẽ tâm của các hình tròn
                cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            plt.figure(figsize=(12, 10))
            plt.title(f"Kết quả phát hiện bubbles: {len(circles[0])} bubbles được tìm thấy")
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return circles[0]
            
        except Exception as e:
            print(f"⚠️ Lỗi khi phát hiện hình tròn: {e}")
            return []
    
    def group_circles_into_questions(self, circles, num_options=4):
        """Nhóm các hình tròn thành các câu hỏi"""
        if len(circles) == 0:
            return []
            
        # Sắp xếp các hình tròn theo tọa độ y (từ trên xuống dưới)
        circles_sorted_by_y = sorted(circles, key=lambda x: x[1])
        
        # Tính khoảng cách trung bình giữa các dòng
        y_distances = []
        for i in range(1, len(circles_sorted_by_y)):
            y_distances.append(circles_sorted_by_y[i][1] - circles_sorted_by_y[i-1][1])
        
        if not y_distances:
            return []
            
        avg_line_height = np.median(y_distances)
        line_threshold = avg_line_height * 0.7  # Ngưỡng để xác định cùng một dòng
        
        # Nhóm các hình tròn theo dòng
        current_line = [circles_sorted_by_y[0]]
        lines = []
        current_y = circles_sorted_by_y[0][1]
        
        for circle in circles_sorted_by_y[1:]:
            if abs(circle[1] - current_y) <= line_threshold:
                current_line.append(circle)
            else:
                # Sắp xếp các hình tròn trong dòng từ trái sang phải
                lines.append(sorted(current_line, key=lambda c: c[0]))
                current_line = [circle]
                current_y = circle[1]
        
        # Thêm dòng cuối cùng
        if current_line:
            lines.append(sorted(current_line, key=lambda c: c[0]))
        
        # Nhóm các hình tròn thành câu hỏi (mỗi câu thường có 4 lựa chọn A, B, C, D)
        questions = []
        
        for line in lines:
            # Tính khoảng cách trung bình giữa các hình tròn trong cùng một dòng
            if len(line) < 2:
                continue
                
            x_distances = []
            for i in range(1, len(line)):
                x_distances.append(line[i][0] - line[i-1][0])
            
            if not x_distances:
                continue
                
            avg_option_distance = np.median(x_distances)
            option_threshold = avg_option_distance * 1.5  # Ngưỡng để xác định các lựa chọn khác nhau
            
            # Nhóm các lựa chọn của cùng một câu hỏi
            current_question = [line[0]]
            question_groups = []
            
            for i in range(1, len(line)):
                if (line[i][0] - line[i-1][0]) <= option_threshold:
                    current_question.append(line[i])
                    # Nếu đã đủ số lượng lựa chọn trong một câu hỏi
                    if len(current_question) == num_options:
                        question_groups.append(current_question)
                        current_question = []
                else:
                    # Nếu khoảng cách quá lớn, đánh dấu kết thúc câu hỏi hiện tại
                    if current_question:
                        question_groups.append(current_question)
                        current_question = [line[i]]
            
            # Thêm nhóm cuối cùng nếu còn
            if current_question:
                question_groups.append(current_question)
            
            # Thêm các câu hỏi đã nhóm vào danh sách kết quả
            questions.extend(question_groups)
        
        return questions
    
    def detect_filled_bubbles(self, original_image, questions):
        """Phát hiện các ô bubble đã được tô"""
        result_image = original_image.copy()
        answers = {}
        
        # Các chữ cái tương ứng với các lựa chọn
        options = ['A', 'B', 'C', 'D']
        
        for q_idx, question in enumerate(questions):
            max_filled = -1
            max_filled_idx = -1
            
            # Hiển thị từng câu hỏi để debug
            question_image = result_image.copy()
            
            for opt_idx, circle in enumerate(question):
                # Trích xuất vùng hình tròn để kiểm tra mức độ tô
                x, y, r = circle
                
                # Added safety checks
                y_start = max(0, y - r)
                y_end = min(original_image.shape[0], y + r)
                x_start = max(0, x - r)
                x_end = min(original_image.shape[1], x + r)
                
                if y_start >= y_end or x_start >= x_end:
                    print(f"⚠️ Tọa độ hình tròn không hợp lệ: x={x}, y={y}, r={r}")
                    continue
                
                roi = original_image[y_start:y_end, x_start:x_end]
                
                if roi.size == 0:
                    print(f"⚠️ Vùng ROI rỗng cho hình tròn: x={x}, y={y}, r={r}")
                    continue
                
                # Chuyển sang ảnh xám nếu là ảnh màu
                if len(roi.shape) == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                
                # Tính mức độ tô đen
                # Đảo ngược giá trị để các vùng đen có giá trị cao hơn
                fill_ratio = 255 - np.mean(roi_gray)
                
                # Vẽ mức độ tô lên ảnh kết quả để debug
                cv2.putText(question_image, f"{fill_ratio:.1f}", (x+r, y-r), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Cập nhật lựa chọn có mức độ tô cao nhất
                if fill_ratio > max_filled:
                    max_filled = fill_ratio
                    max_filled_idx = opt_idx
            
            # Hiển thị câu hỏi với thông tin mức độ tô
            plt.figure(figsize=(8, 6))
            plt.title(f"Câu hỏi {q_idx+1}")
            plt.imshow(cv2.cvtColor(question_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Xác định đáp án được chọn
            # Ngưỡng tối thiểu để coi là được tô (điều chỉnh theo mức độ tô đen của các ô)
            min_fill_threshold = 120
            
            if max_filled >= min_fill_threshold:
                answers[q_idx+1] = options[max_filled_idx]
                
                # Vẽ kết quả lên ảnh
                circle = question[max_filled_idx]
                cv2.circle(result_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 3)
                cv2.putText(result_image, f"Q{q_idx+1}:{options[max_filled_idx]}", 
                            (circle[0] - circle[2], circle[1] - circle[2]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị kết quả cuối cùng
        plt.figure(figsize=(12, 10))
        plt.title("Kết quả nhận dạng đáp án")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return answers
    
    def recognize(self, image_path):
        """Hàm chính để xử lý và nhận dạng phiếu trả lời"""
        # Tiền xử lý ảnh
        binary_image, original_image = self.process_image(image_path)
        
        # Trích xuất thông tin từ form
        form_info = self.extract_fields_from_form(original_image, binary_image)
        
        # Phát hiện các bubble đáp án
        circles = self.detect_answer_bubbles(binary_image, original_image)
        
        # Nhóm các bubble thành câu hỏi
        questions = self.group_circles_into_questions(circles)
        
        # Phát hiện các bubble đã được tô
        answers = self.detect_filled_bubbles(original_image, questions)
        
        # Tổng hợp kết quả
        result = {
            "form_info": form_info,
            "answers": answers,
            "num_questions": len(questions)
        }
        
        return result
    
    def display_results(self, result):
        """Hiển thị kết quả nhận dạng một cách trực quan"""
        # Hiển thị thông tin từ form
        print("=" * 50)
        print("THÔNG TIN PHIẾU TRẢ LỜI")
        print("=" * 50)
        
        form_info = result["form_info"]
        for key, value in form_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Hiển thị đáp án
        print("\n" + "=" * 50)
        print(f"ĐÁP ÁN (Tổng số câu hỏi: {result['num_questions']})")
        print("=" * 50)
        
        answers = result["answers"]
        # Sắp xếp câu hỏi theo thứ tự tăng dần
        for q_idx in sorted(answers.keys()):
            print(f"Câu {q_idx}: {answers[q_idx]}")
        
        # Tính số câu đã trả lời
        num_answered = len(answers)
        num_questions = result["num_questions"]
        
        if num_questions > 0:
            percentage = num_answered/num_questions*100
            print(f"\nSố câu đã trả lời: {num_answered}/{num_questions} ({percentage:.1f}%)")
        else:
            print("\nKhông phát hiện được câu hỏi nào trong phiếu trả lời.")


# Sử dụng ví dụ:
if __name__ == "__main__":
    # Tạo đối tượng nhận dạng
    recognizer = VietnameseAnswerSheetRecognizer(
        character_model_path="btl/cnn.h5",
        encoder_path="btl/cnn-label_encoder_classes.npy",
        force_cnn=False  # Set to True nếu muốn bắt buộc sử dụng CNN thay vì OCR
    )
    
    # Đường dẫn đến ảnh phiếu trả lời cần nhận dạng
    image_path = r"D:\STUDY\DEE\MACHINE LEARNING & APP\character\btl\Screenshot 2025-05-11 182233.png"
    
    # Nhận dạng phiếu
    result = recognizer.recognize(image_path)
    
    # Hiển thị kết quả
    recognizer.display_results(result)