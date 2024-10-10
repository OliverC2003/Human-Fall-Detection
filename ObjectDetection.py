from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') # loading yoloV8 model

path_to_video = r"Coffee_room_01\Coffee_room_01\Videos\video (7).avi"

cap = cv2.VideoCapture(path_to_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret = True
person_class_id = [0]

while ret:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame or video has ended.")
        break

    results = model.track(frame, persist=True, classes=person_class_id)
    annotated_frame = results[0].plot() 
    cv2.imshow('YOLOv8 Tracking', annotated_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
