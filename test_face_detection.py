import os

from video_engine.scene_detector import extract_frames
from video_engine.face_detector import (
    detect_faces,
    extract_faces
)


video_path = "test.mp4"

frame_folder = "frames"
face_folder = "faces"


os.makedirs(
    frame_folder,
    exist_ok=True
)

os.makedirs(
    face_folder,
    exist_ok=True
)


# Extract frames from the video
frames = extract_frames(
    video_path,
    frame_folder,
    interval=1
)


print("\nFACE DETECTION")
print("==============")

total_faces = 0


for frame in frames:

    image_path = frame["path"]

    faces = detect_faces(
        image_path
    )

    extracted = extract_faces(
        image_path,
        face_folder,
        os.path.splitext(
            os.path.basename(image_path)
        )[0]
    )

    total_faces += len(faces)

    print(
        f"{image_path} → "
        f"{len(faces)} face(s)"
    )


print("\n----------------")
print(
    f"Total faces detected: {total_faces}"
)

print(
    f"Face images saved in: {face_folder}/"
)