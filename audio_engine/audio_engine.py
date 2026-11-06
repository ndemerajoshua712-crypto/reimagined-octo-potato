from audio_engine.audio_analyzer import analyze_audio
from audio_engine.spectral_analyzer import analyze_spectrum
from audio_engine.mfcc_analyzer import analyze_mfcc


def analyze_audio_file(audio_path):

    print("Running basic audio analysis...")
    basic = analyze_audio(audio_path)

    print("Running spectral analysis...")
    spectral = analyze_spectrum(audio_path)

    print("Running MFCC analysis...")
    mfcc = analyze_mfcc(audio_path)

    return {
        "basic": basic,
        "spectral": spectral,
        "mfcc": mfcc
    }