import os
import torch
import librosa

from audio_engine.audio_model import AudioDeepfakeModel


# ============================================================
# DEEPTRACE AUDIO MODEL FINAL VALIDATION
# ============================================================

MODEL_PATH = "audio_model/best.pt"

TEST_AUDIO = "audio/test_audio.wav"
REAL_AUDIO = "audio/real_human_fixed.wav"

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64000


print("=" * 60)
print("DEEPTRACE AUDIO MODEL FINAL VALIDATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

print("\n[1] Checking files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(TEST_AUDIO):
    raise FileNotFoundError(
        f"Test audio not found: {TEST_AUDIO}"
    )

if not os.path.exists(REAL_AUDIO):
    raise FileNotFoundError(
        f"Real human audio not found: {REAL_AUDIO}"
    )

print("Model:       OK")
print("Test audio:  OK")
print("Real audio:  OK")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[2] Loading DeepTrace audio model...")

model = AudioDeepfakeModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

checkpoint_state = checkpoint["model_state_dict"]

# Remove training-only parameter
checkpoint_state = {
    key: value
    for key, value in checkpoint_state.items()
    if key != "class_weights"
}

model_state = model.state_dict()

compatible_state = {}
skipped = []

for key, value in checkpoint_state.items():

    if key not in model_state:
        skipped.append(
            f"{key} - unknown parameter"
        )
        continue

    if model_state[key].shape != value.shape:
        skipped.append(
            f"{key} - shape mismatch"
        )
        continue

    compatible_state[key] = value


missing, unexpected = model.load_state_dict(
    compatible_state,
    strict=False
)

model.eval()


print("\nCheckpoint:")
print(
    f"  Parameters in checkpoint: "
    f"{len(checkpoint_state)}"
)

print(
    f"  Parameters loaded: "
    f"{len(compatible_state)}"
)

print(
    f"  Missing parameters: "
    f"{len(missing)}"
)

print(
    f"  Unexpected parameters: "
    f"{len(unexpected)}"
)

if missing:

    print("\nMissing:")

    for key in missing:
        print(
            f"  - {key}"
        )

if skipped:

    print("\nSkipped:")

    for item in skipped:
        print(
            f"  - {item}"
        )


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def prepare_audio(audio_path):

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    original_duration = (
        len(audio) / SAMPLE_RATE
    )

    # Convert to exactly 4 seconds.

    if len(audio) < TARGET_SAMPLES:

        padding = (
            TARGET_SAMPLES - len(audio)
        )

        audio = torch.nn.functional.pad(
            torch.tensor(
                audio,
                dtype=torch.float32
            ),
            (0, padding)
        )

    else:

        audio = torch.tensor(
            audio[:TARGET_SAMPLES],
            dtype=torch.float32
        )

    audio = audio.unsqueeze(0)

    return audio, original_duration


# ============================================================
# MODEL INFERENCE
# ============================================================

def analyze_audio(audio_path, name):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    waveform, duration = prepare_audio(
        audio_path
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Input samples: "
        f"{waveform.shape[1]}"
    )

    with torch.no_grad():

        output = model(waveform)

        logits = output["logits"]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    real_probability = (
        probabilities[0].item()
    )

    spoof_probability = (
        probabilities[1].item()
    )

    predicted_class = torch.argmax(
        probabilities
    ).item()

    if predicted_class == 0:

        prediction = "REAL / BONAFIDE"

    else:

        prediction = "AI / SPOOF"

    print("\nRESULT")

    print(
        f"Real / Bonafide: "
        f"{real_probability * 100:.6f}%"
    )

    print(
        f"AI / Spoof:      "
        f"{spoof_probability * 100:.6f}%"
    )

    print(
        f"Predicted class: "
        f"{predicted_class}"
    )

    print(
        f"Prediction:      "
        f"{prediction}"
    )

    return {
        "name": name,
        "real": real_probability,
        "spoof": spoof_probability,
        "prediction": prediction
    }


# ============================================================
# TEST BOTH FILES
# ============================================================

test_result = analyze_audio(
    TEST_AUDIO,
    "TEST AUDIO"
)

real_result = analyze_audio(
    REAL_AUDIO,
    "REAL HUMAN AUDIO"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"\nTest audio AI probability: "
    f"{test_result['spoof'] * 100:.6f}%"
)

print(
    f"Human audio AI probability: "
    f"{real_result['spoof'] * 100:.6f}%"
)

difference = (
    test_result["spoof"]
    - real_result["spoof"]
)

print(
    f"\nProbability difference: "
    f"{difference * 100:.6f} percentage points"
)


# ============================================================
# VALIDATION INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION STATUS")
print("=" * 60)

if real_result["spoof"] < 0.5:

    print(
        "\nPASS: The model does not classify "
        "the human recording as AI."
    )

else:

    print(
        "\nFAIL: The model still classifies "
        "the human recording as AI."
    )

    print(
        "\nThe problem is NOT the decision threshold."
    )

    print(
        "The checkpoint architecture or "
        "audio preprocessing still needs correction."
    )


print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)