import cv2

from audio_engine.audio_analyzer import analyze_audio
from audio_engine.av_sync_analyzer import analyze_av_sync


video_path = "test.mp4"
audio_path = "audio/test_audio.wav"


# Get video duration
video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

video_duration = frames / fps

video.release()


# Get audio duration
audio_info = analyze_audio(audio_path)

audio_duration = audio_info["duration"]


# Analyze synchronization
result = analyze_av_sync(
    video_duration,
    audio_duration
)


print("\n" + "=" * 60)
print("DEEPTRACE A/V SYNCHRONIZATION TEST")
print("=" * 60)

print(f"\nVideo duration: {result['video_duration']:.2f} seconds")
print(f"Audio duration: {result['audio_duration']:.2f} seconds")
print(
    f"Difference: "
    f"{result['duration_difference']:.2f} seconds"
)

print(
    f"Sync status: "
    f"{result['sync_status']}"
)