from audio_engine.mfcc_analyzer import analyze_mfcc


AUDIO_PATH = "audio/test_audio.wav"


print("\nDEEPTRACE MFCC ANALYZER TEST")
print("============================")

result = analyze_mfcc(AUDIO_PATH)

print("\nMFCC analysis successful!")

print("Sample rate:", result["sample_rate"])
print("Duration:", result["duration"])
print("MFCC coefficients:", result["mfcc_coefficients"])
print("Time frames:", result["time_frames"])

print("\nMean MFCC:")
print(result["mean_mfcc"])

print("\nMFCC standard deviation:")
print(result["std_mfcc"])