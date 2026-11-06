import wave
import numpy as np


def analyze_audio(audio_path):
    """
    Analyze basic characteristics of an extracted WAV file.
    """

    with wave.open(audio_path, "rb") as audio:

        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()

        raw_audio = audio.readframes(frame_count)

    # Convert 16-bit PCM audio to numbers
    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16
    )

    # Normalize to -1.0 to 1.0
    samples = samples.astype(np.float32) / 32768.0

    duration = len(samples) / sample_rate

    # Basic signal measurements
    mean_amplitude = float(np.mean(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration": duration,
        "samples": len(samples),
        "mean_amplitude": mean_amplitude,
        "rms": rms,
        "peak_amplitude": peak
    }