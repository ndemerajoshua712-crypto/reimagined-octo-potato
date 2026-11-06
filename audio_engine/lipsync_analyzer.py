import math

import numpy as np


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

# MediaPipe Face Mesh landmark indices
LIP_UPPER_INNER = 13
LIP_LOWER_INNER = 14
LIP_LEFT_CORNER = 78
LIP_RIGHT_CORNER = 308

SYNC_TOLERANCE_MS = 120      # offsets inside this are treated as in sync
SIGNIFICANT_FLOOR = 0.20     # minimum peak to call the match real
NULL_MARGIN = 0.08           # peak must beat chance-level correlation by this
WEAK_CORR = 0.12             # below this: no relationship at all
NULL_NEAR_S = 1.0            # lags beyond this are used as the chance baseline
MIN_FACE_COVERAGE = 0.25     # fraction of frames with a tracked face
MIN_SPEECH_RATIO = 0.10      # fraction of frames with audible audio
MIN_MOUTH_MOVEMENT = 0.012   # std of raw mouth openness


# ------------------------------------------------------------
# RESULT HELPERS
# ------------------------------------------------------------

def lip_sync_unavailable(reason, status="UNAVAILABLE"):
    """Result with every key the UI reads, for failures or missing data."""
    return {
        "available": False,
        "status": status,
        "reason": reason,
        "lip_sync_score": 0.0,
        "audio_delay_ms": 0.0,
        "peak_correlation": 0.0,
        "confidence": 0.0,
        "chance_level": 0.0,
        "face_coverage": 0.0,
        "speech_ratio": 0.0,
        "fps": 0.0,
        "windows": [],
        "mouth_trace": [],
        "audio_trace": [],
    }


def _classify(peak, lag_ms, null_level=None):
    """
    null_level = how large correlations get by chance in THIS video
    (95th percentile of |correlation| at lags far from zero). The peak
    must clearly stand above it, so noisy videos are judged fairly.
    """
    needed = SIGNIFICANT_FLOOR

    if null_level is not None:
        needed = max(SIGNIFICANT_FLOOR, null_level + NULL_MARGIN)

    if peak >= needed:
        if abs(lag_ms) <= SYNC_TOLERANCE_MS:
            return "SYNCHRONIZED"
        return "OFFSET DETECTED"
    if peak >= WEAK_CORR:
        return "WEAK CORRELATION"
    return "NO CORRELATION"


def _score(peak, lag_ms):
    """0..1, higher = mouth and audio agree more."""
    base = min(1.0, max(0.0, peak) / 0.5)
    if abs(lag_ms) > SYNC_TOLERANCE_MS:
        penalty = max(0.3, 1.0 - (abs(lag_ms) - SYNC_TOLERANCE_MS) / 400.0)
        base *= penalty
    return float(base)


# ------------------------------------------------------------
# SIGNAL EXTRACTION
# ------------------------------------------------------------

def extract_mouth_signal(video_path, max_seconds=90, target_width=480):
    """
    Returns (mouth_openness, fps).
    mouth_openness has one value per video frame, NaN where no face
    was tracked.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the video for lip tracking.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 120:
        fps = 25.0

    max_frames = int(max_seconds * fps)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    values = []

    try:
        while len(values) < max_frames:
            ok, frame = cap.read()

            if not ok:
                break

            height, width = frame.shape[:2]

            if width > target_width:
                scale = target_width / float(width)
                frame = cv2.resize(
                    frame,
                    (target_width, int(height * scale)),
                )
                height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)

            if not result.multi_face_landmarks:
                values.append(np.nan)
                continue

            lm = result.multi_face_landmarks[0].landmark

            def point(index):
                return np.array([lm[index].x * width, lm[index].y * height])

            opening = np.linalg.norm(
                point(LIP_UPPER_INNER) - point(LIP_LOWER_INNER)
            )
            mouth_width = np.linalg.norm(
                point(LIP_LEFT_CORNER) - point(LIP_RIGHT_CORNER)
            )

            if mouth_width < 1e-6:
                values.append(np.nan)
            else:
                values.append(float(opening / mouth_width))

    finally:
        cap.release()
        mesh.close()

    return np.array(values, dtype=np.float64), float(fps)


def _load_wav(path):
    """Mono float32 samples and sample rate."""
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1), int(sr)

    except ImportError:
        pass

    import wave

    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError(
            "Only 16-bit PCM WAV is supported without soundfile installed."
        )

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, int(sr)


def audio_envelope(samples, sample_rate, fps, n_frames):
    """RMS loudness per video frame."""
    hop = sample_rate / fps
    envelope = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = int(round(i * hop))
        end = max(int(round((i + 1) * hop)), start + 1)
        segment = samples[start:end]

        if segment.size:
            envelope[i] = math.sqrt(float(np.mean(segment ** 2)))

    return envelope


# ------------------------------------------------------------
# SIGNAL PROCESSING
# ------------------------------------------------------------

def _fill_short_gaps(signal, max_gap):
    out = signal.copy()
    is_nan = np.isnan(out)
    n = len(out)
    i = 0

    while i < n:
        if is_nan[i]:
            j = i

            while j < n and is_nan[j]:
                j += 1

            if i > 0 and j < n and (j - i) <= max_gap:
                out[i:j] = np.interp(
                    np.arange(i, j),
                    [i - 1, j],
                    [out[i - 1], out[j]],
                )

            i = j
        else:
            i += 1

    return out


def _prepare(signal, fps):
    """Fill short gaps, remove slow trend, smooth, z-score."""
    s = _fill_short_gaps(signal, max_gap=int(0.3 * fps))
    valid = ~np.isnan(s)

    if valid.sum() < 3:
        return s

    filled = np.where(valid, s, np.nanmean(s))

    window = max(3, int(round(fps)))
    trend = np.convolve(filled, np.ones(window) / window, mode="same")

    detrended = filled - trend
    smooth = np.convolve(detrended, np.ones(3) / 3, mode="same")

    spread = np.std(smooth[valid])

    if spread < 1e-9:
        smooth = smooth * 0.0
    else:
        smooth = smooth / spread

    smooth[~valid] = np.nan

    return smooth


def _xcorr(a, b, max_lag, min_overlap):
    """corr[k] = correlation of a[t] with b[t + k]."""
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)

    for i, k in enumerate(lags):
        if k >= 0:
            x, y = a[: n - k], b[k:]
        else:
            x, y = a[-k:], b[: n + k]

        mask = ~(np.isnan(x) | np.isnan(y))

        if mask.sum() < min_overlap:
            continue

        xs, ys = x[mask], y[mask]

        if xs.std() < 1e-6 or ys.std() < 1e-6:
            continue

        out[i] = float(np.corrcoef(xs, ys)[0, 1])

    return lags, out


def _null_level(a, b, fps, min_overlap):
    """95th percentile of |corr| at lags well away from zero (chance level)."""
    n = len(a)
    far_max_s = min(3.0, 0.4 * n / fps)

    if far_max_s <= NULL_NEAR_S + 0.3:
        return None

    far_max = int(round(far_max_s * fps))
    lags, corr = _xcorr(a, b, far_max, min_overlap)
    far = np.abs(lags) >= int(round(NULL_NEAR_S * fps))
    values = np.abs(corr[far])
    values = values[~np.isnan(values)]

    if len(values) < 6:
        return None

    return float(np.percentile(values, 95))


def _best_lag(a, b, fps, max_lag_ms, min_overlap):
    max_lag = max(1, int(round(max_lag_ms / 1000.0 * fps)))
    lags, corr = _xcorr(a, b, max_lag, min_overlap)

    if np.all(np.isnan(corr)):
        return None

    idx = int(np.nanargmax(corr))
    peak = float(corr[idx])
    null_level = _null_level(a, b, fps, min_overlap)
    baseline = null_level if null_level is not None else float(
        np.nanmedian(np.abs(corr))
    )

    return {
        "lag_ms": float(lags[idx] / fps * 1000.0),
        "peak": peak,
        "null_level": null_level,
        "confidence": max(0.0, peak - baseline),
    }


def _windowed(a, b, fps, max_lag_ms, window_s=6.0, hop_s=3.0):
    n = len(a)
    window = int(window_s * fps)
    hop = max(1, int(hop_s * fps))
    results = []

    for number, start in enumerate(range(0, max(1, n - window + 1), hop), 1):
        seg_a = a[start:start + window]
        seg_b = b[start:start + window]

        entry = {
            "window": number,
            "start_time": round(start / fps, 2),
            "end_time": round(min(start + window, n) / fps, 2),
        }

        if np.mean(~np.isnan(seg_a)) < 0.5:
            entry.update(
                {
                    "audio_delay_ms": 0.0,
                    "peak_correlation": 0.0,
                    "status": "NO FACE",
                }
            )
            results.append(entry)
            continue

        best = _best_lag(seg_a, seg_b, fps, max_lag_ms, int(1.5 * fps))

        if best is None:
            entry.update(
                {
                    "audio_delay_ms": 0.0,
                    "peak_correlation": 0.0,
                    "status": "INSUFFICIENT DATA",
                }
            )
        else:
            entry.update(
                {
                    "audio_delay_ms": round(best["lag_ms"], 1),
                    "peak_correlation": round(best["peak"], 3),
                    "status": _classify(
                        best["peak"], best["lag_ms"], best["null_level"]
                    ),
                }
            )

        results.append(entry)

    return results


def _trace(signal):
    return [None if np.isnan(v) else round(float(v), 3) for v in signal]


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def compute_sync(mouth_raw, audio_raw, fps, max_lag_ms=500):
    """Core analysis on two per-frame signals (no video/audio I/O)."""
    n = len(mouth_raw)

    coverage = float(np.mean(~np.isnan(mouth_raw))) if n else 0.0

    p95 = float(np.percentile(audio_raw, 95)) if n else 0.0
    speech_ratio = (
        float(np.mean(audio_raw > 0.25 * p95)) if p95 > 1e-4 else 0.0
    )

    base = lip_sync_unavailable("", status="INSUFFICIENT DATA")
    base.update(
        {
            "fps": round(fps, 2),
            "face_coverage": round(coverage, 3),
            "speech_ratio": round(speech_ratio, 3),
        }
    )

    if n < int(3 * fps):
        base["reason"] = "The video is too short for lip-sync analysis."
        return base

    if coverage < MIN_FACE_COVERAGE:
        base["reason"] = (
            f"A face was tracked in only {coverage * 100:.0f}% of frames."
        )
        return base

    if speech_ratio < MIN_SPEECH_RATIO:
        base["reason"] = "Very little audible speech was found."
        return base

    movement = float(np.nanstd(mouth_raw))

    if movement < MIN_MOUTH_MOVEMENT:
        base["reason"] = "The tracked mouth barely moves."
        return base

    mouth = _prepare(mouth_raw, fps)
    audio = _prepare(audio_raw, fps)

    best = _best_lag(mouth, audio, fps, max_lag_ms, int(1.5 * fps))

    if best is None:
        base["reason"] = "Not enough overlapping data to correlate."
        return base

    status = _classify(best["peak"], best["lag_ms"], best["null_level"])

    reasons = {
        "SYNCHRONIZED": (
            "Mouth movement and audio loudness agree, with an offset of "
            f"{best['lag_ms']:+.0f} ms."
        ),
        "OFFSET DETECTED": (
            "Mouth movement and audio correlate, but the audio is "
            f"shifted by {best['lag_ms']:+.0f} ms."
        ),
        "WEAK CORRELATION": (
            "Mouth movement and audio loudness agree only weakly. "
            "Background noise, music or several speakers can cause this."
        ),
        "NO CORRELATION": (
            "No meaningful relationship was found between mouth movement "
            "and the audio. The speech may not belong to the visible face."
        ),
    }

    base.update(
        {
            "available": True,
            "status": status,
            "reason": (
                reasons[status]
                + (
                    " Short clip (under 10 s): treat this as low reliability."
                    if n / fps < 10.0
                    else ""
                )
                + " (Exploratory heuristic, not proof.)"
            ),
            "lip_sync_score": round(_score(best["peak"], best["lag_ms"]), 3),
            "audio_delay_ms": round(best["lag_ms"], 1),
            "peak_correlation": round(best["peak"], 3),
            "confidence": round(best["confidence"], 3),
            "chance_level": round(best["null_level"] or 0.0, 3),
            "windows": _windowed(mouth, audio, fps, max_lag_ms),
            "mouth_trace": _trace(mouth),
            "audio_trace": _trace(audio),
        }
    )

    return base


def analyze_lip_sync(video_path, audio_path, max_seconds=90, max_lag_ms=500):
    """Full pipeline: video + wav in, lip-sync result dict out."""
    mouth_raw, fps = extract_mouth_signal(video_path, max_seconds)

    if len(mouth_raw) == 0:
        return lip_sync_unavailable("No video frames could be read.")

    samples, sample_rate = _load_wav(audio_path)

    audio_raw = audio_envelope(samples, sample_rate, fps, len(mouth_raw))

    return compute_sync(mouth_raw, audio_raw, fps, max_lag_ms)
