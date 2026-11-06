import hashlib
import json
import shutil
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def probe_video_metadata(path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"available": False, "error": "ffprobe was not found."}

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, OSError) as error:
        return {"available": False, "error": str(error)}

    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"available": False, "error": f"Invalid ffprobe output: {error}"}

    media_format = probe.get("format", {})
    streams = probe.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        {},
    )
    tags = {
        **media_format.get("tags", {}),
        **video.get("tags", {}),
    }
    normalized_tags = {str(key).lower(): value for key, value in tags.items()}

    return {
        "available": True,
        "format": media_format.get("format_name"),
        "duration_seconds": _float_or_none(media_format.get("duration")),
        "bit_rate": media_format.get("bit_rate"),
        "encoder": normalized_tags.get("encoder"),
        "creation_time": normalized_tags.get("creation_time"),
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": audio.get("sample_rate"),
        "audio_channels": audio.get("channels"),
    }


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None