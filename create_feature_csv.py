import pose_estimation
import os

def process_videos_in_directory(path_file):
    """
    Iterates through every video in the given directory and processes it.

    Args:
        path_file (str): Path to the directory containing the videos.
    """
    video_extension = '.avi'

    for file_name in os.listdir(path_file):
        file_path = os.path.join(path_file, file_name)

        if file_name.lower().endswith(video_extension):
            pose_estimation.store_features(file_path)

paths_to_videos = [r"C:\Users\olive\OneDrive\Documents\HFD\Data\Coffee_room_01\Coffee_room_01\Videos", 
                  r"C:\Users\olive\OneDrive\Documents\HFD\Data\Coffee_room_02\Coffee_room_02\Videos",
                  r"C:\Users\olive\OneDrive\Documents\HFD\Data\Home_01\Home_01\Videos",
                  r"C:\Users\olive\OneDrive\Documents\HFD\Data\Home_02\Home_02\Videos"]

for path in paths_to_videos:
    process_videos_in_directory(path)