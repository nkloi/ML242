import cv2
from ultralytics import YOLO


model = YOLO("best.pt")  


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)


if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model.predict(source=frame, conf=0.7, save=False)

    
    has_detection = False

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            has_detection = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


   
    cv2.imshow("Waste Detection (YOLO)", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
