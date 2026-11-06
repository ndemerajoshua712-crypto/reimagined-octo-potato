from multimodal_evidence_scorer import calculate_multimodal_score


video_score = 0.4032
temporal_consistency = 0.9227
audio_score = 0.9970
av_sync_score = 1.0


result = calculate_multimodal_score(
    video_score,
    temporal_consistency,
    audio_score,
    av_sync_score
)


print("=" * 60)
print("DEEPTRACE MULTIMODAL EVIDENCE TEST")
print("=" * 60)

print(f"Video evidence: {video_score * 100:.2f}%")
print(f"Temporal consistency: {temporal_consistency * 100:.2f}%")
print(f"Audio evidence: {audio_score * 100:.2f}%")
print(f"A/V sync evidence: {av_sync_score * 100:.2f}%")

print(f"\nFINAL SCORE: {result['final_percentage']:.2f}%")
print(f"VERDICT: {result['verdict']}")