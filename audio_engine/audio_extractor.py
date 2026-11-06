import os
import subprocess


def extract_audio(video_path, output_path):
    """
    Extract the audio track from a video.

    Output:
        WAV audio, 16 kHz, mono
    """

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Audio extraction failed:\n"
            + result.stderr
        )

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            "Audio file was not created."
        )

    return output_path