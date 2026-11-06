from audio_engine.audio_model import AudioDeepfakeModel
from audio_engine.audio_window_analyzer import analyze_audio_windows


def test_file(model, path):

    print("\n" + "=" * 60)
    print(f"Testing: {path}")
    print("=" * 60)

    result = analyze_audio_windows(
        path,
        model
    )

    print(f"\nAudio duration: {result['audio_duration']:.2f} seconds")
    print(f"Total windows: {result['total_windows']}")
    print(f"Window size: {result['window_seconds']} seconds")
    print(f"Step size: {result['step_seconds']} seconds")

    print("\nOVERALL RESULT")
    print("-" * 60)

    print(
        f"Average AI/Spoof score: "
        f"{result['average_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"Maximum AI/Spoof score: "
        f"{result['maximum_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"Minimum AI/Spoof score: "
        f"{result['minimum_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"AI/Spoof windows: "
        f"{result['spoof_windows']} / "
        f"{result['total_windows']}"
    )

    print(
        f"AI/Spoof window percentage: "
        f"{result['spoof_window_percentage']:.2f}%"
    )

    print(
        f"Overall prediction: "
        f"{result['overall_prediction']}"
    )

    print("\nWINDOW RESULTS")
    print("-" * 60)

    for window in result["windows"]:

        print(
            f"Window {window['window']}: "
            f"{window['start_time']:.1f}s - "
            f"{window['end_time']:.1f}s | "
            f"Spoof: "
            f"{window['spoof_probability'] * 100:.2f}% | "
            f"{window['prediction']}"
        )


model = AudioDeepfakeModel()

print("\nLoading AASIST-L...")
model.load_model()

test_file(
    model,
    "audio/test_audio.wav"
)

test_file(
    model,
    "audio/real_human_fixed.wav"
)