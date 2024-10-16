import cv2
import mediapipe as mp
import os
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def run_pose_estimation(video_path):
    """
    Function to perform simplified pose estimation on a given video file.

    Args:
        video_path (str): Path to the video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        if results.pose_landmarks:
            important_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_INDEX,
                mp_pose.PoseLandmark.LEFT_INDEX
            ]

            for landmark in important_landmarks:
                lm = results.pose_landmarks.landmark[landmark]
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

            connections = [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
                           (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                           (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                           (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                           (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                           (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                           (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                           (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                           (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                           (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_INDEX),
                           (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_INDEX)]

            for start, end in connections:
                start_point = results.pose_landmarks.landmark[start]
                end_point = results.pose_landmarks.landmark[end]
                cv2.line(frame, (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0])),
                              (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0])), (255, 0, 0), 2)

        cv2.imshow('Simplified Pose Estimation', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def store_features(video_path):
    """
    Function to store features (landmark coordinates) of important pose landmarks at each frame.

    Args:
        video_path (str): Path to the video file.
    """
    video_dir = os.path.dirname(video_path)
    features_dir = os.path.join(video_dir, 'features')
    
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    csv_file_path = os.path.join(features_dir, 'pose_landmarks.csv')
    
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['frame', 'landmark', 'x', 'y', 'z'])

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = 0

        important_landmarks = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.LEFT_INDEX
        ]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks:
                for landmark in important_landmarks:
                    lm = results.pose_landmarks.landmark[landmark]
                    x, y, z = lm.x, lm.y, lm.z
                    csvwriter.writerow([frame_count, landmark.name, x, y, z])

            frame_count += 1

        cap.release()

