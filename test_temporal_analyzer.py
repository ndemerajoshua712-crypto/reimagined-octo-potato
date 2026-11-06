from video_engine.scene_detector import extract_frames
from video_engine.face_pipeline import analyze_frames
from video_engine.temporal_analyzer import (
    analyze_temporal_consistency
)


VIDEO_PATH = "test.mp4"

FRAME_FOLDER = "frames"

FACE_FOLDER = "faces"


print("\nDEEPTRACE TEMPORAL ANALYSIS TEST")
print("================================")


print("\nExtracting frames...")

frames = extract_frames(
    VIDEO_PATH,
    FRAME_FOLDER,
    interval=0.2
)


print("Analyzing faces...")

results = analyze_frames(
    frames,
    face_folder=FACE_FOLDER
)


print("\nAnalyzing temporal consistency...")

temporal = analyze_temporal_consistency(
    results
)


print("\nTEMPORAL REPORT")
print("===============")

print(
    "Frames analyzed:",
    temporal["frames_analyzed"]
)

print(
    "Mean fake probability:",
    f"{temporal['mean_fake_probability'] * 100:.2f}%"
)

print(
    "Average deviation:",
    f"{temporal['average_deviation'] * 100:.2f}%"
)

print(
    "Consistency:",
    f"{temporal['consistency'] * 100:.2f}%"
)


print("\nFrame scores:")

for frame in temporal["frame_scores"]:

    print(
        f"Time {frame['timestamp']:.2f}s → "
        f"{frame['fake_probability'] * 100:.2f}% fake"
    )