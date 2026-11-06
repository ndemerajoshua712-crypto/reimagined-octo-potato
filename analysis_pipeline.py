from pathlib import Path

import cv2
import numpy as np

from audio_engine.audio_analyzer import analyze_audio
from audio_engine.audio_evidence_scorer import score_audio_evidence
from audio_engine.audio_extractor import extract_audio
from audio_engine.lipsync_analyzer import analyze_lip_sync, lip_sync_unavailable
from audio_engine.audio_model import AudioDeepfakeModel
from audio_engine.audio_window_analyzer import analyze_audio_windows
from audio_engine.av_sync_analyzer import analyze_av_sync
from calibration import apply_calibration, load_calibration
from forensic_report import generate_forensic_report
from multimodal_evidence_scorer import calculate_multimodal_score
from video_engine.face_pipeline import analyze_frames
from video_engine.provenance import probe_video_metadata, sha256_file
from video_engine.scene_detector import detect_scenes, extract_frames
from video_engine.temporal_analyzer import analyze_temporal_consistency


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
        "Prototype visual animation/CGI heuristic; not proof of AI generation."
        if detected
        else "Visual characteristics were not strong enough for the prototype heuristic."
    )
    return {"detected": detected, "confidence": min(1.0, score + 0.4) if detected else score, "reason": reason}


def analyze_video_file(
    video_path,
    filename,
    frame_interval=0.5,
    output_root=None,
    progress_callback=None,
):
    def report_progress(stage, progress):
        if progress_callback is not None:
            progress_callback(stage, progress)

    report_progress("provenance", 8)
    video_path = Path(video_path).resolve()
    output_root = Path(output_root or Path(__file__).resolve().parent)
    file_hash = sha256_file(video_path)
    case_id = file_hash[:16]
    provenance = probe_video_metadata(video_path)
    case_dir = output_root / "cases" / case_id
    frame_dir = case_dir / "frames"
    face_dir = case_dir / "faces"
    audio_dir = case_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    scenes = detect_scenes(str(video_path))
    report_progress("scene_detection", 15)
    frames = extract_frames(str(video_path), str(frame_dir), interval=frame_interval)
    if not frames:
        raise RuntimeError("No frames could be extracted from the video.")

    report_progress("face_analysis", 25)
    video_results = analyze_frames(frames, face_folder=str(face_dir))
    report_progress("temporal_analysis", 52)
    try:
        temporal_result = analyze_temporal_consistency(video_results)
        temporal_consistency = temporal_result["temporal_consistency"]
        video_score = temporal_result["mean_fake_probability"]
    except ValueError:
        temporal_result = {
            "frames_analyzed": len(video_results),
            "faces_analyzed": 0,
            "mean_fake_probability": 0.0,
            "average_deviation": 0.0,
            "temporal_consistency": 0.0,
            "frame_scores": [],
        }
        temporal_consistency = 0.0
        video_score = 0.0

    audio_path = audio_dir / "audio.wav"
    extract_audio(str(video_path), str(audio_path))
    report_progress("audio_analysis", 62)
    audio_model = AudioDeepfakeModel()
    audio_model.load_model()
    audio_windows = analyze_audio_windows(str(audio_path), audio_model)
    audio_evidence = score_audio_evidence(audio_windows)
    audio_info = analyze_audio(str(audio_path))
    report_progress("av_analysis", 82)

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps if fps > 0 else 0.0
    capture.release()
    sync_result = analyze_av_sync(video_duration, audio_info["duration"])
    try:
        lip_sync = analyze_lip_sync(str(video_path), str(audio_path))
    except Exception as error:
        lip_sync = lip_sync_unavailable(str(error))
    audio_score = audio_evidence["evidence_score"]
    multimodal_result = calculate_multimodal_score(
        video_score,
        audio_score,
        temporal_consistency,
        sync_result["sync_status"],
    )
    calibration_artifact = load_calibration()
    calibrated_primary_score = apply_calibration(
        multimodal_result["primary_score"], calibration_artifact
    )
    calibration_info = None
    if calibration_artifact is not None:
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
            "mean_fake_probability": video_score,
            "frames_analyzed": temporal_result["frames_analyzed"],
            "faces_analyzed": temporal_result["faces_analyzed"],
        },
        audio_result={
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
        "animation_result": animation_result,
        "report": report,
        "frame_interval": frame_interval,
        "syncnet": {
            "available": False,
            "status": "No compatible SyncNet model code or pretrained checkpoint is configured.",
        },
    }
