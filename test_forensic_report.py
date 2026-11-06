from forensic_report import generate_forensic_report
from multimodal_evidence_scorer import calculate_multimodal_score


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

video_score = 0.4032
audio_score = 0.9970
temporal_consistency = 0.9227
av_sync_status = "SIGNIFICANT OFFSET"


# --------------------------------------------------
# MULTIMODAL ANALYSIS
# --------------------------------------------------

multimodal_result = calculate_multimodal_score(
    video_score=video_score,
    audio_score=audio_score,
    temporal_consistency=temporal_consistency,
    av_sync_status=av_sync_status
)


# --------------------------------------------------
# VIDEO DATA
# --------------------------------------------------

video_result = {
    "mean_fake_probability": video_score,
    "frames_analyzed": 160,
    "faces_analyzed": 160
}


# --------------------------------------------------
# AUDIO DATA
# --------------------------------------------------

audio_result = {
    "average_spoof_probability": 0.9970,
    "maximum_spoof_probability": 1.0,
    "spoof_window_percentage": 100.0
}


# --------------------------------------------------
# TEMPORAL DATA
# --------------------------------------------------

temporal_result = {
    "temporal_consistency": temporal_consistency,
    "average_deviation": 0.0773
}


# --------------------------------------------------
# SYNC DATA
# --------------------------------------------------

sync_result = {
    "video_duration": 31.17,
    "audio_duration": 30.09,
    "duration_difference": 1.08,
    "sync_status": av_sync_status
}


# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------

report = generate_forensic_report(
    filename="test.mp4",
    multimodal_result=multimodal_result,
    video_result=video_result,
    audio_result=audio_result,
    temporal_result=temporal_result,
    sync_result=sync_result
)


# --------------------------------------------------
# DISPLAY REPORT
# --------------------------------------------------

print(report)