from ultralytics import YOLO
import os
from PIL import Image


model = YOLO('best.pt') 

# Folder paths
image_folder = 'dataset/images/train/metal'
label_folder = 'dataset/labels/train/metal'
os.makedirs(label_folder, exist_ok=True)

# Run inference on all images in the folder
for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(image_folder, img_file)
        
        # Run detection
        results = model(img_path)[0]

        # Get image size for normalization
        with Image.open(img_path) as img:
            width, height = img.size

        # Prepare .txt file
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                x_center, y_center, w, h = box.xywhn[0]  # normalized [0,1]
                f.write(f"{2} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        print(f"Labeled: {img_file}")
