from audio_engine.audio_analyzer import analyze_audio


AUDIO_PATH = "audio/test_audio.wav"


print("\nDEEPTRACE AUDIO ANALYZER TEST")
print("==============================")

result = analyze_audio(AUDIO_PATH)

print("\nAudio analysis successful!")

for key, value in result.items():
    print(f"{key}: {value}")