import os

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, threshold=27.0):
    """
    Detect scene changes in a video.

    Returns:
        List of scenes as:
        [
            {
                "scene": 1,
                "start": start_time,
                "end": end_time
            }
        ]
    """

    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold)
    )

    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    scenes = []

    for number, (start, end) in enumerate(scene_list, start=1):
        scenes.append({
            "scene": number,
            "start": start.get_seconds(),
            "end": end.get_seconds()
        })

    return scenes


def extract_frames(video_path, output_folder, interval=1):
    """
    Extract one frame every `interval` seconds.
    """

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_interval = int(fps * interval)

    frame_number = 0
    saved_frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            timestamp = frame_number / fps

            filename = (
                f"{output_folder}/frame_{timestamp:.2f}.jpg"
            )

            if not cv2.imwrite(filename, frame):
                cap.release()
                raise OSError(f"Could not write extracted frame: {filename}")

            saved_frames.append({
                "timestamp": timestamp,
                "path": filename
            })

        frame_number += 1

    cap.release()

    return saved_frames