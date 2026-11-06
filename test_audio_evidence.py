from audio_engine.audio_model import AudioDeepfakeModel
from audio_engine.audio_window_analyzer import analyze_audio_windows
from audio_engine.audio_evidence_scorer import score_audio_evidence


def test_file(model, path):

    print("\n" + "=" * 60)
    print(path)
    print("=" * 60)

    windows = analyze_audio_windows(
        path,
        model
    )

    evidence = score_audio_evidence(
        windows
    )

    print(
        f"Average spoof score: "
        f"{windows['average_spoof_probability'] * 100:.2f}%"
    )

    print(
        f"Suspicious windows: "
        f"{windows['spoof_window_percentage']:.2f}%"
    )

    print(
        f"Evidence score: "
        f"{evidence['evidence_percentage']:.2f}%"
    )

    print(
        f"Evidence level: "
        f"{evidence['evidence_level']}"
    )


model = AudioDeepfakeModel()
model.load_model()

test_file(model, "audio/test_audio.wav")
test_file(model, "audio/real_human_fixed.wav")