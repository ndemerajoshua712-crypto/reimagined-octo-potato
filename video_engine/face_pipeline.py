import os
from pathlib import Path

from video_engine.face_detector import (
    detect_faces,
    extract_faces
)

from video_engine.deepfake_detector import (
    DeepfakeDetector
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "DeepfakeBench"
    / "training"
    / "weights"
    / "xception_best.pth"
)


def analyze_frames(
    frame_list,
    face_folder="faces",
    model_path=DEFAULT_MODEL_PATH
):

    os.makedirs(face_folder, exist_ok=True)

    # Create detector
    detector = DeepfakeDetector(model_path)

    print("Loading deepfake detection model...")
    detector.load_model()
    print("Deepfake model loaded.")

    results = []

    for frame in frame_list:

        image_path = frame["path"]

        frame_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        # Detect faces
        faces = detect_faces(image_path)

        # Extract faces
        extracted_faces = extract_faces(
            image_path,
            face_folder,
            frame_name,
            detected_faces=faces,
        )

        # Analyze every extracted face
        for face in extracted_faces:

            detection = detector.analyze_face(
                face["path"]
            )

            face["prediction"] = detection["label"]
            face["confidence"] = detection["confidence"]
            face["real_probability"] = detection[
                "real_probability"
            ]
            face["fake_probability"] = detection[
                "fake_probability"
            ]

        results.append({
            "frame": image_path,
            "timestamp": frame.get("timestamp"),
            "face_count": len(faces),
            "faces": extracted_faces
        })

    return results