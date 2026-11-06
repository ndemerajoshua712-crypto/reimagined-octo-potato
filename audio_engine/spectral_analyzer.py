import wave
import numpy as np


def analyze_spectrum(audio_path):
    """
    Analyze the frequency spectrum of a WAV audio file.
    """

    with wave.open(audio_path, "rb") as audio:

        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()

        raw_audio = audio.readframes(frame_count)

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16
    ).astype(np.float32)

    samples = samples / 32768.0

    # Use a section of the audio for FFT analysis
    segment_length = min(sample_rate, len(samples))

    segment = samples[:segment_length]

    # Apply a Hann window
    window = np.hanning(len(segment))
    windowed = segment * window

    # Fast Fourier Transform
    spectrum = np.fft.rfft(windowed)

    magnitudes = np.abs(spectrum)

    frequencies = np.fft.rfftfreq(
        len(segment),
        1 / sample_rate
    )

    # Ignore DC component
    magnitudes[0] = 0

    dominant_index = np.argmax(magnitudes)

    dominant_frequency = float(
        frequencies[dominant_index]
    )

    spectral_energy = float(
        np.sum(magnitudes ** 2)
    )

    return {
        "sample_rate": sample_rate,
        "dominant_frequency": dominant_frequency,
        "spectral_energy": spectral_energy,
        "frequency_bins": len(frequencies)
    }