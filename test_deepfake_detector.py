from video_engine.deepfake_detector import DeepfakeDetector


MODEL_PATH = "DeepfakeBench/training/weights/xception_best.pth"


detector = DeepfakeDetector(MODEL_PATH)

print("\nDEEPTRACE DEEPFAKE DETECTOR TEST")
print("================================")

print("Loading model...")
detector.load_model()

print("Model loaded successfully!")


faces = [
    "faces/frame_0.98_face_1.jpg",
    "faces/frame_0.98_face_2.jpg"
]


for face in faces:

    print("\nAnalyzing:", face)

    result = detector.analyze_face(face)

    print("Prediction:", result["label"])
    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )
    print(
        "REAL probability:",
        f"{result['real_probability'] * 100:.2f}%"
    )
    print(
        "FAKE probability:",
        f"{result['fake_probability'] * 100:.2f}%"
    )