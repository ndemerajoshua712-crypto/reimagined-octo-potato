from audio_engine.audio_engine import analyze_audio_file


AUDIO_PATH = "audio/test_audio.wav"


print("\nDEEPTRACE AUDIO ENGINE TEST")
print("============================")

result = analyze_audio_file(AUDIO_PATH)

print("\nAudio engine completed successfully!")

print("\n--- BASIC AUDIO ---")
print("Duration:", result["basic"]["duration"])
print("RMS:", result["basic"]["rms"])
print("Peak:", result["basic"]["peak_amplitude"])

print("\n--- SPECTRAL ---")
print(
    "Dominant frequency:",
    result["spectral"]["dominant_frequency"]
)

print(
    "Spectral energy:",
    result["spectral"]["spectral_energy"]
)

print("\n--- MFCC ---")
print(
    "MFCC coefficients:",
    result["mfcc"]["mfcc_coefficients"]
)

print(
    "Time frames:",
    result["mfcc"]["time_frames"]
)