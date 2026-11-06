import os
import sys
import numpy as np
import torch

# Allow Python to find the official AASIST network
AASIST_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "audio_model",
    "aasist"
)

if AASIST_FOLDER not in sys.path:
    sys.path.insert(0, AASIST_FOLDER)

from _net import Model as AASISTNet


class AudioDeepfakeModel:

    SAMPLE_RATE = 16000
    TARGET_SAMPLES = 64600

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint_path = os.path.join(
            AASIST_FOLDER,
            "AASIST-L.pth"
        )

    def load_model(self):

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"AASIST checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 24],
                [24, 24]
            ],
            "gat_dims": [24, 32],
            "pool_ratios": [0.4, 0.5, 0.7, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }

        print("Loading AASIST-L...")

        self.model = AASISTNet(config)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get(
                "state_dict",
                checkpoint
            )

        self.model.load_state_dict(
            checkpoint,
            strict=True
        )

        self.model.eval()
        self.model.to(self.device)

        print("AASIST-L loaded successfully.")
        print(f"Device: {self.device}")

    def prepare_audio(self, waveform):

        waveform = np.asarray(
            waveform,
            dtype=np.float32
        ).reshape(-1)

        if len(waveform) >= self.TARGET_SAMPLES:

            waveform = waveform[
                :self.TARGET_SAMPLES
            ]

        else:

            repetitions = (
                self.TARGET_SAMPLES // len(waveform)
            ) + 1

            waveform = np.tile(
                waveform,
                repetitions
            )[:self.TARGET_SAMPLES]

        return waveform.astype(np.float32)

    def predict(self, waveform):

        if self.model is None:
            raise RuntimeError(
                "AASIST-L has not been loaded."
            )

        waveform = self.prepare_audio(
            waveform
        )

        audio_tensor = torch.from_numpy(
            waveform
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            _, logits = self.model(
                audio_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        # Official AASIST mapping:
        # class 0 = spoof
        # class 1 = bona fide

        spoof_probability = (
            probabilities[0].item()
        )

        real_probability = (
            probabilities[1].item()
        )

        if real_probability >= spoof_probability:

            prediction = "REAL / BONAFIDE"
            confidence = real_probability

        else:

            prediction = "AI / SPOOF"
            confidence = spoof_probability

        return {
            "prediction": prediction,
            "confidence": confidence,
            "real_probability": real_probability,
            "spoof_probability": spoof_probability
        }
