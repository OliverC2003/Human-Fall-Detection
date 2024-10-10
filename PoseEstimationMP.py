import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

path_to_video = r"C:\Users\olive\OneDrive\Documents\HFD\Data\Coffee_room_01\Coffee_room_01\Videos\video (1).avi"

cap = cv2.VideoCapture(path_to_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Pose Estimation', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
