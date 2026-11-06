import torch

from audio_engine.audio_model import AudioDeepfakeModel
from audio_engine.audio_window_analyzer import analyze_audio_windows


MODEL_PATH = "audio_model/best.pt"
TEST_AUDIO = "audio/test_audio.wav"
REAL_AUDIO = "audio/real_human_fixed.wav"


print("======================================")
print("DEEPTRACE AUDIO MODEL VALIDATION")
print("======================================")


# ======================================
# LOAD MODEL
# ======================================

print("\nLoading model...")

model = AudioDeepfakeModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

state = checkpoint["model_state_dict"]

# Remove training-only parameter
state = {
    key: value
    for key, value in state.items()
    if key != "class_weights"
}


# ======================================
# LOAD COMPATIBLE PARAMETERS
# ======================================

model_state = model.state_dict()
compatible_state = {}

for key, value in state.items():

    if key in model_state:

        if model_state[key].shape == value.shape:

            compatible_state[key] = value

        else:

            print(
                f"Skipping shape mismatch: {key}"
            )

    else:

        print(
            f"Skipping unknown checkpoint key: {key}"
        )


missing, unexpected = model.load_state_dict(
    compatible_state,
    strict=False
)

model.eval()


print(
    f"\nLoaded {len(compatible_state)} "
    "trained parameters."
)

print(
    f"Missing model parameters: "
    f"{len(missing)}"
)

print(
    f"Unexpected parameters: "
    f"{len(unexpected)}"
)


# ======================================
# AUDIO TEST FUNCTION
# ======================================

def test_audio(audio_path, name):

    print("\n======================================")
    print(name)
    print("======================================")

    try:

        analysis = analyze_audio_windows(
            audio_path,
            model
        )

    except FileNotFoundError:

        print(
            f"ERROR: File not found:\n"
            f"{audio_path}"
        )

        return

    except Exception as error:

        print(
            f"ERROR analyzing audio:\n"
            f"{error}"
        )

        return


    # ==================================
    # RESULTS
    # ==================================

    print(
        f"\nDuration: "
        f"{analysis['audio_duration']:.2f} seconds"
    )

    print(
        f"Windows analyzed: "
        f"{analysis['total_windows']}"
    )

    print(
        f"Average AI probability: "
        f"{analysis['average_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"Minimum AI probability: "
        f"{analysis['minimum_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"Maximum AI probability: "
        f"{analysis['maximum_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"AI probability variation: "
        f"{analysis['spoof_probability_std'] * 100:.2f}%"
    )

    print(
        f"AI windows: "
        f"{analysis['spoof_windows']} / "
        f"{analysis['total_windows']}"
    )

    print(
        f"AI window percentage: "
        f"{analysis['spoof_window_percentage']:.2f}%"
    )

    print(
        f"Final model prediction: "
        f"{analysis['overall_prediction']}"
    )


    # ==================================
    # INDIVIDUAL WINDOWS
    # ==================================

    print("\nWindow Details:")

    for window in analysis["windows"]:

        print(
            f"Window {window['window']:02d} | "
            f"{window['start_time']:.1f}s - "
            f"{window['end_time']:.1f}s | "
            f"Real: "
            f"{window['real_probability'] * 100:.2f}% | "
            f"AI: "
            f"{window['spoof_probability'] * 100:.2f}% | "
            f"{window['prediction']}"
        )


# ======================================
# TEST ORIGINAL AUDIO
# ======================================

test_audio(
    TEST_AUDIO,
    "TEST AUDIO"
)


# ======================================
# TEST REAL HUMAN AUDIO
# ======================================

test_audio(
    REAL_AUDIO,
    "REAL HUMAN AUDIO"
)


# ======================================
# FINISHED
# ======================================

print("\n======================================")
print("VALIDATION COMPLETE")
print("======================================")