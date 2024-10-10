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
        h, w, _ = frame.shape
        
        # Draw all landmarks except for facial details
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i in [mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]:
                continue  # Skip nose and wrists for custom drawing later
            else:
                # Get landmark coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # Small white circles for body landmarks
        
        # Custom drawing for head and hands
        # Get nose (head)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)  # Green circle for head (nose)

        # Get left wrist
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_wrist_x = int(left_wrist.x * w)
        left_wrist_y = int(left_wrist.y * h)
        cv2.circle(frame, (left_wrist_x, left_wrist_y), 5, (255, 0, 0), -1)  # Blue circle for left hand (wrist)

        # Get right wrist
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_wrist_x = int(right_wrist.x * w)
        right_wrist_y = int(right_wrist.y * h)
        cv2.circle(frame, (right_wrist_x, right_wrist_y), 5, (255, 0, 0), -1)  # Blue circle for right hand (wrist)

    cv2.imshow('MediaPipe Pose Estimation', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
