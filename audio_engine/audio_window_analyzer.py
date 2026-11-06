import librosa
import numpy as np


SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0


def analyze_audio_windows(
    audio_path,
    model,
    window_seconds=WINDOW_SECONDS,
    step_seconds=STEP_SECONDS
):

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file contains no audio.")

    window_samples = int(
        window_seconds * sample_rate
    )

    step_samples = int(
        step_seconds * sample_rate
    )

    if len(audio) < sample_rate:
        raise ValueError(
            "Audio is too short for window analysis."
        )

    results = []

    start = 0
    window_number = 1

    while start < len(audio):

        end = start + window_samples

        segment = audio[
            start:min(end, len(audio))
        ]

        # Ignore a tiny leftover section
        if len(segment) < sample_rate:
            break

        # AASIST-L handles the exact input length
        result = model.predict(segment)

        results.append({

            "window": window_number,

            "start_time": round(
                start / sample_rate,
                3
            ),

            "end_time": round(
                min(end, len(audio)) / sample_rate,
                3
            ),

            "real_probability":
                result["real_probability"],

            "spoof_probability":
                result["spoof_probability"],

            "prediction":
                result["prediction"],

            "confidence":
                result["confidence"]
        })

        start += step_samples
        window_number += 1

    if not results:
        raise ValueError(
            "No usable audio windows were created."
        )

    spoof_scores = np.array([
        result["spoof_probability"]
        for result in results
    ])

    spoof_windows = sum(
        1
        for result in results
        if result["prediction"] == "AI / SPOOF"
    )

    spoof_window_percentage = (
        spoof_windows / len(results)
    ) * 100

    average_spoof = float(
        np.mean(spoof_scores)
    )

    if average_spoof >= 0.5:
        overall_prediction = "AI / SPOOF"
    else:
        overall_prediction = "REAL / BONAFIDE"

    return {

        "audio_duration":
            len(audio) / sample_rate,

        "sample_rate":
            sample_rate,

        "total_windows":
            len(results),

        "window_seconds":
            window_seconds,

        "step_seconds":
            step_seconds,

        "average_spoof_probability":
            average_spoof,

        "maximum_spoof_probability":
            float(np.max(spoof_scores)),

        "minimum_spoof_probability":
            float(np.min(spoof_scores)),

        "spoof_probability_std":
            float(np.std(spoof_scores)),

        "spoof_windows":
            spoof_windows,

        "spoof_window_percentage":
            float(spoof_window_percentage),

        "overall_prediction":
            overall_prediction,

        "overall_spoof_score":
            average_spoof,

        "windows":
            results
    }