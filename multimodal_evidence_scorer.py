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

    # Primary model evidence
    video_evidence = video_score * 0.50
    audio_evidence = audio_score * 0.50

    # Primary multimodal score
    primary_score = (
        video_evidence +
        audio_evidence
    )

    # Supporting indicators
    if temporal_consistency >= 0.90:
        temporal_status = "STABLE"
    elif temporal_consistency >= 0.75:
        temporal_status = "MODERATE"
    else:
        temporal_status = "UNSTABLE"

    if av_sync_status == "GOOD":
        sync_status = "GOOD"
    elif av_sync_status == "MINOR OFFSET":
        sync_status = "MINOR OFFSET"
    else:
        sync_status = "SIGNIFICANT OFFSET"

    # Verdict based primarily on model evidence
    if primary_score >= 0.75:
        verdict = "HIGH AI/DEEPFAKE EVIDENCE"
    elif primary_score >= 0.55:
        verdict = "MODERATE AI/DEEPFAKE EVIDENCE"
    elif primary_score >= 0.40:
        verdict = "LOW AI/DEEPFAKE EVIDENCE"
    else:
        verdict = "VERY LOW AI/DEEPFAKE EVIDENCE"

    return {
        "video_evidence": video_score,
        "audio_evidence": audio_score,
        "primary_score": primary_score,
        "temporal_consistency": temporal_consistency,
        "temporal_status": temporal_status,
        "av_sync_status": sync_status,
        "verdict": verdict
    }