from video_engine.scene_detector import detect_scenes, extract_frames
import os

video_path = "test.mp4"
output_folder = "frames"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Detect scenes
scenes = detect_scenes(video_path)

print("\nDetected Scenes:")
print("----------------")

for scene in scenes:
    print(
        f"Scene {scene['scene']}: "
        f"{scene['start']:.2f}s → "
        f"{scene['end']:.2f}s"
    )

# Extract frames
print("\nExtracting frames...")
print("--------------------")

frames = extract_frames(
    video_path,
    output_folder,
    interval=1
)

print(f"\nExtracted {len(frames)} frames.")

for frame in frames[:10]:
    print(
        f"Frame: {frame['path']} "
        f"| Time: {frame['timestamp']:.2f}s"
    )