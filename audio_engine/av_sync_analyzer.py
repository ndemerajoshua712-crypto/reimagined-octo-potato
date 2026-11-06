def analyze_av_sync(video_duration, audio_duration):

    if audio_duration is None or audio_duration <= 0:
        return {
            "video_duration": float(video_duration or 0),
            "audio_duration": 0.0,
            "duration_difference": 0.0,
            "sync_status": "UNAVAILABLE",
        }

    difference = abs(
        video_duration - audio_duration
    )

    if difference <= 0.10:
        status = "GOOD"
    elif difference <= 0.50:
        status = "MINOR OFFSET"
    else:
        status = "SIGNIFICANT OFFSET"

    return {
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "duration_difference": difference,
        "sync_status": status
    }
