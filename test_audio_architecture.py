from audio_engine.audio_model import AudioDeepfakeModel

print("Loading Wav2Vec2 architecture...")

model = AudioDeepfakeModel()

print("Architecture loaded successfully.")
print(model.classifier)