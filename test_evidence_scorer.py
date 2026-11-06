from video_engine.scene_detector import extract_frames
from video_engine.face_pipeline import analyze_frames
from video_engine.evidence_scorer import calculate_evidence_score


VIDEO_PATH = "test.mp4"

FRAME_FOLDER = "frames"

FACE_FOLDER = "faces"


print("\nDEEPTRACE EVIDENCE SCORING TEST")
print("================================")


print("\nExtracting frames...")

frames = extract_frames(
    VIDEO_PATH,
    FRAME_FOLDER,
    interval=1
)


print("Analyzing faces...")

results = analyze_frames(
    frames,
    face_folder=FACE_FOLDER
)


print("\nCalculating evidence score...")

score = calculate_evidence_score(
    results
)


print("\nEVIDENCE REPORT")
print("===============")

print(
    "Evidence score:",
    f"{score['evidence_score']:.2f}%"
)

print(
    "Faces analyzed:",
    score["faces_analyzed"]
)

print(
    "Risk level:",
    score["risk_level"]
)

print(
    "Status:",
    score["status"]
)