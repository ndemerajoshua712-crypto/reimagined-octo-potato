from audio_engine.audio_extractor import extract_audio


VIDEO_PATH = "test.mp4"

AUDIO_PATH = "audio/test_audio.wav"


print("\nDEEPTRACE AUDIO EXTRACTION TEST")
print("================================")

print("Extracting audio...")

output = extract_audio(
    VIDEO_PATH,
    AUDIO_PATH
)

print("\nAudio extraction successful!")

print("Output:")
print(output)