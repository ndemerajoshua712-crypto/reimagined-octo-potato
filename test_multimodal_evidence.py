from multimodal_evidence_scorer import calculate_multimodal_score


video_score = 0.4032
audio_score = 0.9970

temporal_consistency = 0.9227

av_sync_status = "SIGNIFICANT OFFSET"


result = calculate_multimodal_score(
    video_score=video_score,
    audio_score=audio_score,
    temporal_consistency=temporal_consistency,
    av_sync_status=av_sync_status
)


print("=" * 60)
print("DEEPTRACE MULTIMODAL EVIDENCE TEST")
print("=" * 60)

print(
    f"\nVideo evidence: "
    f"{result['video_evidence'] * 100:.2f}%"
)

print(
    f"Audio evidence: "
    f"{result['audio_evidence'] * 100:.2f}%"
)

print(
    f"\nPrimary evidence score: "
    f"{result['primary_score'] * 100:.2f}%"
)

print(
    f"Temporal consistency: "
    f"{result['temporal_consistency'] * 100:.2f}%"
)

print(
    f"Temporal status: "
    f"{result['temporal_status']}"
)

print(
    f"A/V synchronization: "
    f"{result['av_sync_status']}"
)

print(
    f"\nVERDICT: "
    f"{result['verdict']}"
)