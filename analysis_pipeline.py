import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from audio_engine.audio_evidence_scorer import score_audio_evidence
from audio_engine.audio_extractor import extract_audio
from audio_engine.lipsync_analyzer import analyze_lip_sync, lip_sync_unavailable
from audio_engine.audio_model import AASIST_FOLDER, AudioDeepfakeModel
from audio_engine.audio_window_analyzer import analyze_audio_windows
from audio_engine.av_sync_analyzer import analyze_av_sync
from calibration import apply_calibration, load_calibration
from forensic_report import generate_forensic_report
from multimodal_evidence_scorer import calculate_multimodal_score
from video_engine.face_pipeline import DEFAULT_MODEL_PATH, analyze_frames
from video_engine.provenance import probe_video_metadata, sha256_file
from video_engine.scene_detector import detect_scenes, extract_frames
from video_engine.temporal_analyzer import analyze_temporal_consistency


def _positive_limit(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_ANALYSIS_FRAMES = _positive_limit("DEEPTRACE_MAX_ANALYSIS_FRAMES", 600)
MAX_AUDIO_WINDOWS = _positive_limit("DEEPTRACE_MAX_AUDIO_WINDOWS", 300)
AUDIO_CHECKPOINT_PATH = Path(AASIST_FOLDER) / "AASIST-L.pth"


@lru_cache(maxsize=1)
def _load_audio_model():
    model = AudioDeepfakeModel()
    model.load_model()
    return model


@lru_cache(maxsize=8)
def _checkpoint_sha256(path, size_bytes, modified_ns, file_id):
    return sha256_file(path)


def _checkpoint_info(name, path):
    path = Path(path)
    if not path.is_file():
        return {"name": name, "available": False, "sha256": None}
    stat = path.stat()
    return {
        "name": name,
        "available": True,
        "size_bytes": stat.st_size,
        "sha256": _checkpoint_sha256(
            str(path.resolve()), stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", 0)
        ),
    }


def _empty_audio_result(reason):
    return {
        "available": False,
        "unavailable_reason": reason,
        "audio_duration": 0.0,
        "sample_rate": 16000,
        "total_windows": 0,
        "average_spoof_probability": 0.0,
        "maximum_spoof_probability": 0.0,
        "minimum_spoof_probability": 0.0,
        "spoof_probability_std": 0.0,
        "spoof_windows": 0,
        "spoof_window_percentage": 0.0,
        "overall_prediction": "UNAVAILABLE",
        "overall_spoof_score": 0.0,
        "window_seconds": 4.0,
        "step_seconds": 2.0,
        "sampling_limited": False,
        "windows": [],
    }


def _animation_heuristic(frames, face_results):
    if not frames:
        return {"detected": False, "confidence": 0.0, "reason": "No frames available."}
    if sum(item.get("face_count", 0) for item in face_results):
        return {"detected": False, "confidence": 0.0, "reason": "Human faces were detected."}

    scores = []
    for item in frames[:20]:
        image = cv2.imread(item["path"])
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = float(hsv[:, :, 1].mean() / 255.0)
        edge_density = float(np.mean(cv2.Canny(gray, 100, 200) > 0))
        scores.append(saturation * 0.55 + edge_density * 0.45)

    if not scores:
        return {"detected": False, "confidence": 0.0, "reason": "Visual characteristics could not be measured."}
    score = float(np.mean(scores))
    detected = score >= 0.45
    reason = (
        "Rule-based visual animation/CGI heuristic; not proof of AI generation."
        if detected
        else "Visual characteristics were not strong enough for the appearance heuristic."
    )
    return {"detected": detected, "confidence": min(1.0, score + 0.4) if detected else score, "reason": reason}


def analyze_video_file(
    video_path,
    filename,
    frame_interval=0.5,
    output_root=None,
    progress_callback=None,
    source_sha256=None,
):
    def report_progress(stage, progress):
        if progress_callback is not None:
            progress_callback(stage, progress)

    report_progress("provenance", 8)
    video_path = Path(video_path).resolve()
    output_root = Path(output_root or Path(__file__).resolve().parent)
    file_hash = source_sha256 or sha256_file(video_path)
    case_id = file_hash[:16]
    provenance = probe_video_metadata(video_path)
    model_checkpoints = {
        "video": _checkpoint_info("Xception", DEFAULT_MODEL_PATH),
        "audio": _checkpoint_info("AASIST-L", AUDIO_CHECKPOINT_PATH),
    }
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = (
        frame_count / fps
        if fps > 0
        else float(provenance.get("duration_seconds") or 0.0)
    )
    capture.release()
    if not provenance.get("duration_seconds") and video_duration > 0:
        provenance["duration_seconds"] = video_duration

    requested_frame_interval = max(0.05, float(frame_interval))
    effective_frame_interval = requested_frame_interval
    if video_duration > 0:
        effective_frame_interval = max(
            requested_frame_interval,
            video_duration / MAX_ANALYSIS_FRAMES,
        )
    analysis_warnings = []
    if effective_frame_interval > requested_frame_interval:
        analysis_warnings.append(
            f"Video sampling was capped at about {MAX_ANALYSIS_FRAMES} frames; "
            f"the interval increased from {requested_frame_interval:g}s to "
            f"{effective_frame_interval:.2f}s."
        )
    case_dir = output_root / "cases" / case_id
    frame_dir = case_dir / "frames"
    face_dir = case_dir / "faces"
    audio_dir = case_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    report_progress("scene_detection", 12)
    scenes = detect_scenes(str(video_path))
    report_progress("frame_extraction", 18)
    frames = extract_frames(
        str(video_path), str(frame_dir), interval=effective_frame_interval
    )
    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")

    report_progress("face_analysis", 25)
    video_results = analyze_frames(frames, face_folder=str(face_dir))
    report_progress("temporal_analysis", 52)
    try:
        temporal_result = analyze_temporal_consistency(video_results)
        temporal_consistency = (
            temporal_result["temporal_consistency"]
            if temporal_result["faces_analyzed"] > 0
            else None
        )
        video_score = (
            temporal_result["mean_fake_probability"]
            if temporal_result["faces_analyzed"] > 0
            else None
        )
    except ValueError:
        temporal_result = {
            "frames_analyzed": len(video_results),
            "faces_analyzed": 0,
            "mean_fake_probability": 0.0,
            "average_deviation": 0.0,
            "temporal_consistency": 0.0,
            "frame_scores": [],
        }
        temporal_consistency = None
        video_score = None
        analysis_warnings.append(
            "No usable face predictions were available; the video score is omitted."
        )
    if video_score is None and not any("face predictions" in item for item in analysis_warnings):
        analysis_warnings.append(
            "No usable face predictions were available; the video score is omitted."
        )

    audio_path = audio_dir / "audio.wav"
    report_progress("audio_analysis", 62)
    audio_reason = None
    if provenance.get("available") and not provenance.get("audio_codec"):
        audio_reason = "No audio track was found in the video."
    else:
        try:
            extract_audio(str(video_path), str(audio_path))
            audio_model = _load_audio_model()
            audio_windows = analyze_audio_windows(
                str(audio_path), audio_model, max_windows=MAX_AUDIO_WINDOWS
            )
            audio_windows["available"] = True
            audio_evidence = score_audio_evidence(audio_windows)
            audio_evidence["available"] = True
            audio_info = {"duration": audio_windows["audio_duration"]}
        except Exception as error:
            audio_reason = (
                "The audio track could not be analyzed. Check the worker log for details."
            )
            audio_windows = _empty_audio_result(audio_reason)
            audio_evidence = {
                "available": False,
                "evidence_score": None,
                "evidence_percentage": None,
                "evidence_level": "UNAVAILABLE",
            }
            audio_info = {"duration": 0.0}
            logging.getLogger("deeptrace.analysis").exception(
                "audio analysis unavailable: %s", error
            )
    if audio_reason:
        audio_windows = _empty_audio_result(audio_reason)
        audio_evidence = {
            "available": False,
            "evidence_score": None,
            "evidence_percentage": None,
            "evidence_level": "UNAVAILABLE",
        }
        audio_info = {"duration": 0.0}
        analysis_warnings.append(audio_reason)
    if audio_windows.get("sampling_limited"):
        analysis_warnings.append(
            f"Audio analysis was capped at about {MAX_AUDIO_WINDOWS} windows; "
            f"the effective window step is {audio_windows['step_seconds']:.1f}s."
        )
    report_progress("av_analysis", 82)
    sync_result = analyze_av_sync(video_duration, audio_info["duration"])
    if audio_info["duration"] > 0:
        try:
            lip_sync = analyze_lip_sync(str(video_path), str(audio_path))
        except Exception as error:
            lip_sync = lip_sync_unavailable(str(error))
    else:
        lip_sync = lip_sync_unavailable(audio_reason or "Audio track is unavailable.")
    audio_score = audio_evidence["evidence_score"]
    multimodal_result = calculate_multimodal_score(
        video_score,
        audio_score,
        temporal_consistency,
        sync_result["sync_status"],
    )
    calibration_artifact = load_calibration()
    both_modalities_available = len(multimodal_result["available_modalities"]) == 2
    calibrated_primary_score = (
        apply_calibration(multimodal_result["primary_score"], calibration_artifact)
        if calibration_artifact is not None
        and multimodal_result["primary_score"] is not None
        and both_modalities_available else None
    )
    if calibration_artifact is not None and not both_modalities_available:
        analysis_warnings.append(
            "Calibration was not applied because the saved mapping is intended "
            "for results with both video and audio signals."
        )
    calibration_info = None
    if calibration_artifact is not None and calibrated_primary_score is not None:
        calibration_info = {
            "method": calibration_artifact.get("method", "isotonic_regression"),
            "source_sha256": calibration_artifact.get("source_sha256"),
            "train_rows": calibration_artifact.get("train_rows"),
            "test_rows": calibration_artifact.get("test_rows"),
            "raw_brier": calibration_artifact.get("raw_brier"),
            "calibrated_brier": calibration_artifact.get("calibrated_brier"),
            "calibrated_ece": calibration_artifact.get("calibrated_ece"),
            "calibrated_auc": calibration_artifact.get("calibrated_auc"),
            "subgroup_metrics": calibration_artifact.get("subgroup_metrics", {}),
        }
    animation_result = _animation_heuristic(frames, video_results)
    report_progress("report_generation", 94)

    report = generate_forensic_report(
        filename=filename,
        multimodal_result=multimodal_result,
        video_result={
            "available": video_score is not None,
            "mean_fake_probability": video_score,
            "frames_analyzed": temporal_result["frames_analyzed"],
            "faces_analyzed": temporal_result["faces_analyzed"],
        },
        audio_result={
            "available": audio_windows["available"],
            "unavailable_reason": audio_windows["unavailable_reason"],
            "average_spoof_probability": audio_windows["average_spoof_probability"],
            "maximum_spoof_probability": audio_windows["maximum_spoof_probability"],
            "spoof_window_percentage": audio_windows["spoof_window_percentage"],
        },
        temporal_result={
            "temporal_consistency": temporal_consistency,
            "average_deviation": temporal_result["average_deviation"],
        },
        sync_result=sync_result,
        calibrated_score=calibrated_primary_score,
        calibration_info=calibration_info,
        analysis_warnings=analysis_warnings,
        model_checkpoints=model_checkpoints,
    )
    report_progress("complete", 100)

    return {
        "case_id": case_id,
        "filename": filename,
        "file_hash": file_hash,
        "provenance": provenance,
        "scenes": scenes,
        "frames": frames,
        "video_results": video_results,
        "temporal_result": temporal_result,
        "audio_windows": audio_windows,
        "audio_evidence": audio_evidence,
        "sync_result": sync_result,
        "lip_sync": lip_sync,
        "multimodal_result": multimodal_result,
        "calibrated_primary_score": calibrated_primary_score,
        "calibration": calibration_info,
        "model_checkpoints": model_checkpoints,
        "analysis_warnings": analysis_warnings,
        "animation_result": animation_result,
        "report": report,
        "frame_interval": effective_frame_interval,
        "requested_frame_interval": requested_frame_interval,
        "syncnet": {
            "available": False,
            "status": "No compatible SyncNet model code or pretrained checkpoint is configured.",
        },
    }
