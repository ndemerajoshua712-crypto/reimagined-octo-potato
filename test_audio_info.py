import wave

AUDIO_PATH = "audio/test_audio.wav"

print("\nDEEPTRACE AUDIO INFO TEST")
print("==========================")

with wave.open(AUDIO_PATH, "rb") as audio:

    channels = audio.getnchannels()
    sample_width = audio.getsampwidth()
    sample_rate = audio.getframerate()
    frames = audio.getnframes()
    duration = frames / sample_rate

print("\nAudio information:")
print("Channels:", channels)
print("Sample width:", sample_width, "bytes")
print("Sample rate:", sample_rate, "Hz")
print("Frames:", frames)
print("Duration:", round(duration, 2), "seconds")