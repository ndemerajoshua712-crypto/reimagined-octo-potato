import librosa
import numpy as np


def analyze_mfcc(audio_path):
    """
    Extract MFCC features from an audio file.
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mean_mfcc = np.mean(
        mfcc,
        axis=1
    )

    std_mfcc = np.std(
        mfcc,
        axis=1
    )

    return {
        "sample_rate": sample_rate,
        "duration": len(audio) / sample_rate,
        "mfcc_coefficients": mfcc.shape[0],
        "time_frames": mfcc.shape[1],
        "mean_mfcc": mean_mfcc.tolist(),
        "std_mfcc": std_mfcc.tolist()
    }