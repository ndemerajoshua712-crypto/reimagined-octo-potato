import os

from video_engine.face_preprocessor import (
    preprocess_face,
    prepare_for_model
)


face_folder = "faces"

face_files = [
    os.path.join(face_folder, file)
    for file in os.listdir(face_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("\nDEEPTRACE FACE PREPROCESSOR")
print("==========================")

if not face_files:
    print("No face images found.")
else:

    face_path = face_files[0]

    image = preprocess_face(face_path)

    model_input = prepare_for_model(face_path)

    print("Face:", face_path)
    print("Processed shape:", image.shape)
    print("Model input shape:", model_input.shape)
    print("Data type:", model_input.dtype)
    print("Minimum pixel value:", model_input.min())
    print("Maximum pixel value:", model_input.max())

    print("\nPreprocessing successful.")