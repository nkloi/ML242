from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


model = YOLO("best.pt")  


image_path = "img4398.jpg"
image = cv2.imread(image_path)

results = model(image)


annotated_frame = results[0].plot()

plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("YOLO Detection")
plt.show()
