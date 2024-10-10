import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

path_to_video = r"Coffee_room_01\Coffee_room_01\Videos\video (5).avi"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to read frame or video has ended.")
        break
    
    results = model(frame)

    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
