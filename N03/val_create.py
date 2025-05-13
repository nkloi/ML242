import os
import shutil
import random


dataset_path = 'dataset'  
val_ratio = 0.25  


val_img_path = os.path.join(dataset_path, 'images/val/plastic')
val_lbl_path = os.path.join(dataset_path, 'labels/val/plastic')
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(val_lbl_path, exist_ok=True)


train_img_path = os.path.join(dataset_path, 'images/train/plastic')
train_lbl_path = os.path.join(dataset_path, 'labels/train/plastic')

image_files = [
    f for f in os.listdir(train_img_path)
    if f.endswith(('.jpg', '.jpeg', '.png'))
]


val_count = int(len(image_files) * val_ratio)
val_images = random.sample(image_files, val_count)


for img_file in val_images:
    label_file = os.path.splitext(img_file)[0] + '.txt'

    
    src_img = os.path.join(train_img_path, img_file)
    src_lbl = os.path.join(train_lbl_path, label_file)

    
    dst_img = os.path.join(val_img_path, img_file)
    dst_lbl = os.path.join(val_lbl_path, label_file)

   
    shutil.copyfile(src_img, dst_img)
    if os.path.exists(src_lbl):
        shutil.copyfile(src_lbl, dst_lbl)

print(f"✅ Đã sao chép {val_count} ảnh vào tập validation.")
