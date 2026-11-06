import numpy as np


def analyze_temporal_consistency(results):

    if not results:
        raise ValueError("No video results provided.")

    frame_scores = []

    for frame in results:

        face_scores = []

        for face in frame.get("faces", []):
            if "fake_probability" in face:
                face_scores.append(
                    face["fake_probability"]
                )

        if face_scores:
            frame_score = float(np.mean(face_scores))

            frame_scores.append({
                "frame": frame.get("frame"),
                "timestamp": frame.get("timestamp"),
                "fake_probability": frame_score
            })

    if not frame_scores:
        raise ValueError("No face predictions found.")

    scores = np.array([
        item["fake_probability"]
        for item in frame_scores
    ])

    mean_score = float(np.mean(scores))
    average_deviation = float(np.std(scores))

    consistency = max(
        0.0,
        1.0 - average_deviation
    )

    return {
        "frames_analyzed": len(results),
        "faces_analyzed": len(scores),

        "mean_fake_probability": mean_score,

        "average_deviation": average_deviation,

        "consistency": consistency,

        "temporal_consistency": consistency,

        "frame_scores": frame_scores
    }