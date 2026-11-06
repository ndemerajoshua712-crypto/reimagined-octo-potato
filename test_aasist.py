import librosa

from audio_engine.audio_model import AudioDeepfakeModel


def test_file(model, path):

    print("\n" + "=" * 60)
    print(f"Testing: {path}")
    print("=" * 60)

    audio, sample_rate = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    result = model.predict(audio)

    print("\nRESULT")
    print("-" * 60)

    print(
        f"Real / Bonafide: "
        f"{result['real_probability'] * 100:.4f}%"
    )

    print(
        f"AI / Spoof: "
        f"{result['spoof_probability'] * 100:.4f}%"
    )

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.4f}%"
    )

    return result


model = AudioDeepfakeModel()

model.load_model()

test_file(
    model,
    "audio/test_audio.wav"
)

test_file(
    model,
    "audio/real_human_fixed.wav"
)