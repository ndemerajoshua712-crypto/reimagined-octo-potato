from video_engine.scene_detector import extract_frames
from video_engine.face_pipeline import analyze_frames


video_path = "test.mp4"

frames = extract_frames(
    video_path,
    "frames",
    interval=1
)

results = analyze_frames(
    frames,
    "faces"
)

print("\nDEEPTRACE FACE PIPELINE")
print("=======================")

total_faces = 0

for result in results:

    print(
        f"{result['frame']} → "
        f"{result['face_count']} face(s)"
    )

    total_faces += result["face_count"]

print("\n-----------------------")
print(f"Frames analysed: {len(results)}")
print(f"Total faces detected: {total_faces}")
print("Face extraction complete.")