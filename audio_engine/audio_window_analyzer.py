import librosa
import numpy as np


SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0


def analyze_audio_windows(
    audio_path,
    model,
    window_seconds=WINDOW_SECONDS,
    step_seconds=STEP_SECONDS,
    max_windows=300,
):

    window_seconds = float(window_seconds)
    step_seconds = float(step_seconds)
    max_windows = int(max_windows)
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Audio window and step durations must be positive.")
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1.")

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

    step_samples = max(1, int(step_seconds * sample_rate))

    # Bound model inference on long recordings while preserving samples near
    # both the beginning and end of the audio track.
    max_start = max(0, len(audio) - window_samples)
    estimated_windows = int(np.ceil(max_start / step_samples)) + 1
    sampling_limited = estimated_windows > max_windows
    if max_start == 0:
        starts = [0]
    elif sampling_limited:
        starts = np.linspace(0, max_start, num=max_windows, dtype=np.int64).tolist()
    else:
        starts = list(range(0, len(audio), step_samples))

    if len(audio) < sample_rate:
        raise ValueError(
            "Audio is too short for window analysis."
        )

    results = []

    window_number = 1
    for start in starts:
        end = start + window_samples

        segment = audio[
            start:min(end, len(audio))
        ]

        # Ignore a tiny leftover section
        if len(segment) < sample_rate:
            continue

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
            (starts[1] - starts[0]) / sample_rate if len(starts) > 1 else step_seconds,

        "requested_step_seconds":
            step_seconds,

        "sampling_limited":
            sampling_limited,

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
