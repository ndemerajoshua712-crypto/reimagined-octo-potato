import numpy as np


def score_audio_evidence(window_result):

    average = window_result["average_spoof_probability"]
    maximum = window_result["maximum_spoof_probability"]
    percentage = window_result["spoof_window_percentage"]

    # Combine multiple pieces of evidence
    evidence_score = (
        (average * 0.50) +
        (maximum * 0.20) +
        ((percentage / 100) * 0.30)
    )

    if evidence_score >= 0.75:
        level = "HIGH"
    elif evidence_score >= 0.55:
        level = "MODERATE"
    elif evidence_score >= 0.40:
        level = "LOW"
    else:
        level = "VERY LOW"

    return {
        "evidence_score": float(evidence_score),
        "evidence_percentage": float(evidence_score * 100),
        "evidence_level": level
    }