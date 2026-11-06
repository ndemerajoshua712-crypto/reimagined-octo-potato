from video_engine.scene_detector import (
    extract_frames
)

from video_engine.face_pipeline import (
    analyze_frames
)


VIDEO_PATH = "test.mp4"

FRAME_FOLDER = "frames"

FACE_FOLDER = "faces"


print("\nDEEPTRACE FULL PIPELINE TEST")
print("============================")


# -----------------------------
# 1. Extract frames
# -----------------------------

print("\n[1] Extracting frames...")

frames = extract_frames(
    VIDEO_PATH,
    FRAME_FOLDER,
    interval=1
)

print("Frames extracted:", len(frames))


# -----------------------------
# 2. Detect faces
# -----------------------------

print("\n[2] Detecting and analyzing faces...")

results = analyze_frames(
    frames,
    face_folder=FACE_FOLDER
)


# -----------------------------
# 3. Display results
# -----------------------------

print("\n[3] RESULTS")
print("===========")


total_faces = 0

for frame in results:

    timestamp = frame["timestamp"]

    for face in frame["faces"]:

        total_faces += 1

        print(
            f"\nTime: {timestamp:.2f}s"
        )

        print(
            f"Face: {face['face']}"
        )

        print(
            f"Prediction: {face['prediction']}"
        )

        print(
            f"REAL: "
            f"{face['real_probability'] * 100:.2f}%"
        )

        print(
            f"FAKE: "
            f"{face['fake_probability'] * 100:.2f}%"
        )


print("\n============================")
print("TOTAL FACES ANALYZED:", total_faces)
print("============================")