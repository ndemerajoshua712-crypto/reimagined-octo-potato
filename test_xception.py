import sys
import torch

sys.path.insert(0, "DeepfakeBench/training")

from networks.xception import Xception


MODEL_PATH = (
    "DeepfakeBench/training/weights/xception_best.pth"
)

print("\nDEEPTRACE XCEPTION DETECTOR TEST")
print("================================")

config = {
    "num_classes": 2,
    "mode": "original",
    "inc": 3,
    "dropout": False
}

print("Creating Xception backbone...")

model = Xception(config)

print("Loading trained detector checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# Get the state dictionary
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Remove the "backbone." prefix.
# The checkpoint was saved from DeepfakeBench's
# XceptionDetector, while our model is only Xception.
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

print("\nCHECKPOINT RESULTS")
print("==================")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("\nMissing keys:")
    for key in missing:
        print(" -", key)

if unexpected:
    print("\nUnexpected keys:")
    for key in unexpected:
        print(" -", key)

model.eval()

print("\nModel loading complete.")
print("Device: CPU")