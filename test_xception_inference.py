import sys
import torch
import cv2
import numpy as np

sys.path.insert(0, "DeepfakeBench/training")

from networks.xception import Xception


MODEL_PATH = "DeepfakeBench/training/weights/xception_best.pth"
FACE_PATH = "faces/frame_0.98_face_1.jpg"


print("\nDEEPTRACE XCEPTION INFERENCE TEST")
print("=================================")


# -----------------------------
# 1. Create model
# -----------------------------

config = {
    "num_classes": 2,
    "mode": "original",
    "inc": 3,
    "dropout": False
}

print("Creating Xception model...")
model = Xception(config)


# -----------------------------
# 2. Load trained checkpoint
# -----------------------------

print("Loading trained checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint


clean_state_dict = {}

for name, weights in state_dict.items():

    if name.startswith("backbone."):
        name = name[len("backbone."):]

    if "pointwise" in name and weights.ndim == 2:
        weights = weights.unsqueeze(-1).unsqueeze(-1)

    clean_state_dict[name] = weights


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


if missing or unexpected:
    print("ERROR: checkpoint did not load correctly.")
    sys.exit()


model.eval()


# -----------------------------
# 3. Load face
# -----------------------------

print("\nLoading face:")
print(FACE_PATH)

image = cv2.imread(FACE_PATH)

if image is None:
    print("ERROR: Could not load face image.")
    sys.exit()


# BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Resize to DeepfakeBench resolution
image = cv2.resize(image, (256, 256))


# Convert to float
image = image.astype(np.float32) / 255.0


# Normalize using DeepfakeBench settings
image = (image - 0.5) / 0.5


# HWC -> CHW
image = np.transpose(image, (2, 0, 1))


# Add batch dimension
image = np.expand_dims(image, axis=0)


# NumPy -> PyTorch
tensor = torch.from_numpy(image).float()


print("Input shape:", tensor.shape)


# -----------------------------
# 4. Run inference
# -----------------------------

print("\nRunning AI detection...")

with torch.no_grad():

    output = model(tensor)

    if isinstance(output, tuple):
        output = output[0]


print("Raw model output:")
print(output)


# -----------------------------
# 5. Convert output to scores
# -----------------------------

probabilities = torch.softmax(output, dim=1)

print("\nProbabilities:")
print(probabilities)


predicted_class = torch.argmax(probabilities, dim=1).item()

print("\nPredicted class:", predicted_class)

print("\nINFERENCE COMPLETE")