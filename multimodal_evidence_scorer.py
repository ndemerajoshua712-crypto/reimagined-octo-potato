def calculate_multimodal_score(
    video_score,
    audio_score,
    temporal_consistency,
    av_sync_status
):
    """
    Combine model evidence from video and audio.

    Temporal consistency and A/V synchronization are treated
    as supporting indicators, not direct proof of deepfake content.
    """

    # Missing audio/video is unknown evidence, not a zero-valued clean signal.
    available = {}
    if video_score is not None:
        available["video"] = float(video_score)
    if audio_score is not None:
        available["audio"] = float(audio_score)
    if available:
        weight = 1.0 / len(available)
        fusion_weights = {name: weight for name in available}
        primary_score = sum(value * fusion_weights[name] for name, value in available.items())
    else:
        fusion_weights = {}
        primary_score = None

    # Supporting indicators
    if temporal_consistency is None:
        temporal_status = "UNAVAILABLE"
    elif temporal_consistency >= 0.90:
        temporal_status = "STABLE"
    elif temporal_consistency >= 0.75:
        temporal_status = "MODERATE"
    else:
        temporal_status = "UNSTABLE"

    if av_sync_status == "UNAVAILABLE":
        sync_status = "UNAVAILABLE"
    elif av_sync_status == "GOOD":
        sync_status = "GOOD"
    elif av_sync_status == "MINOR OFFSET":
        sync_status = "MINOR OFFSET"
    else:
        sync_status = "SIGNIFICANT OFFSET"

    # Verdict based primarily on model evidence
    if primary_score is None:
        verdict = "INSUFFICIENT EVIDENCE"
    elif primary_score >= 0.75:
        verdict = "HIGH AI/DEEPFAKE EVIDENCE"
    elif primary_score >= 0.55:
        verdict = "MODERATE AI/DEEPFAKE EVIDENCE"
    elif primary_score >= 0.40:
        verdict = "LOW AI/DEEPFAKE EVIDENCE"
    else:
        verdict = "VERY LOW AI/DEEPFAKE EVIDENCE"
    if len(available) == 1:
        verdict = f"SINGLE-MODAL · {verdict}"

    return {
        "video_evidence": video_score,
        "audio_evidence": audio_score,
        "primary_score": primary_score,
        "available_modalities": list(available),
        "fusion_weights": fusion_weights,
        "temporal_consistency": temporal_consistency,
        "temporal_status": temporal_status,
        "av_sync_status": sync_status,
        "verdict": verdict
    }
