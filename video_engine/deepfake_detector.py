import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

training_path = Path(__file__).resolve().parents[1] / "DeepfakeBench" / "training"
sys.path.insert(0, str(training_path))

from networks.xception import Xception


class DeepfakeDetector:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False

    def load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        config = {
            "num_classes": 2,
            "mode": "original",
            "inc": 3,
            "dropout": False
        }

        self.model = Xception(config)

        checkpoint = torch.load(
            self.model_path,
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

        missing, unexpected = self.model.load_state_dict(
            clean_state_dict,
            strict=False
        )

        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint loading failed.\n"
                f"Missing: {len(missing)}\n"
                f"Unexpected: {len(unexpected)}"
            )

        self.model.eval()
        self.model_loaded = True

    def analyze_face(self, face_image_path):

        if not self.model_loaded:
            raise RuntimeError(
                "Model is not loaded."
            )

        image = cv2.imread(face_image_path)

        if image is None:
            raise ValueError(
                f"Could not read face image: {face_image_path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # DeepfakeBench uses 256x256
        image = cv2.resize(
            image,
            (256, 256)
        )

        # Convert to 0-1
        image = image.astype(
            np.float32
        ) / 255.0

        # Normalize using DeepfakeBench settings
        image = (image - 0.5) / 0.5

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        tensor = torch.from_numpy(
            image
        ).float()

        # Run model
        with torch.no_grad():

            output = self.model(tensor)

            if isinstance(output, tuple):
                output = output[0]

            probabilities = torch.softmax(
                output,
                dim=1
            )

        real_probability = probabilities[0][0].item()
        fake_probability = probabilities[0][1].item()

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        if predicted_class == 1:
            label = "FAKE"
            confidence = fake_probability
        else:
            label = "REAL"
            confidence = real_probability

        return {
            "status": "success",
            "label": label,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability
        }