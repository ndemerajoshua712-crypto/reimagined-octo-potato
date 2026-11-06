import os
import json
import time
import tempfile
import io
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from timeline_chart import build_timeline_figure
from ui_components import (
    inject_css,
    scan_panel,
    TerminalLog,
    render_gauge,
    frame_viewer,
)
from video_engine.scene_detector import detect_scenes, extract_frames
from video_engine.face_pipeline import analyze_frames
from video_engine.provenance import probe_video_metadata, sha256_file
from video_engine.ela import error_level_analysis
from video_engine.temporal_analyzer import analyze_temporal_consistency
from analysis_pipeline import analyze_video_file
from calibration import (
    apply_calibration,
    evaluate_calibration_csv,
    load_calibration,
    save_calibration,
)
from case_history import list_cases, save_case

from audio_engine.audio_extractor import extract_audio
from audio_engine.audio_window_analyzer import analyze_audio_windows
from audio_engine.audio_model import AudioDeepfakeModel
from audio_engine.audio_evidence_scorer import score_audio_evidence
from audio_engine.audio_analyzer import analyze_audio
from audio_engine.av_sync_analyzer import analyze_av_sync
from audio_engine.lipsync_analyzer import (
    analyze_lip_sync,
    lip_sync_unavailable,
)

from multimodal_evidence_scorer import calculate_multimodal_score
from forensic_report import generate_forensic_report


def html(markup: str):
    cleaned = "\n".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


class _TabRenderer:
    def __init__(self, container, streamlit_module):
        self._container = container
        self._streamlit_module = streamlit_module

    def __getattr__(self, name):
        if name == "session_state":
            return self._streamlit_module.session_state
        return getattr(self._container, name)


def store_analysis_result(result):
    st.session_state.analysis_complete = True
    for key in (
        "filename",
        "file_hash",
        "provenance",
        "scenes",
        "frames",
        "video_results",
        "temporal_result",
        "audio_windows",
        "audio_evidence",
        "sync_result",
        "lip_sync",
        "multimodal_result",
        "animation_result",
        "report",
        "frame_interval",
        "syncnet",
    ):
        if key in result:
            st.session_state[key] = result[key]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HUGO | DeepTrace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 229, 255, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(110, 60, 255, 0.08),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #03060b 0%,
                #07101a 45%,
                #03060b 100%
            );
        color: #edf7ff;
    }

    /* ======================================================
       GLITCH TECH BACKGROUND
       ====================================================== */

    .glitch-bg {
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        overflow: hidden;
    }

    .glitch-bg::before {
        content: "GLITCH TECH";
        position: absolute;
        right: -160px;
        top: 25%;
        transform: rotate(-90deg);
        font-size: 110px;
        font-weight: 900;
        letter-spacing: 18px;
        color: rgba(0, 229, 255, 0.025);
        white-space: nowrap;
    }

    .glitch-bg::after {
        content: "G T";
        position: absolute;
        left: 30px;
        bottom: 30px;
        font-size: 180px;
        font-weight: 900;
        letter-spacing: 25px;
        color: rgba(110, 60, 255, 0.025);
    }

    /* ======================================================
       HUGO BRAND
       ====================================================== */

    .hugo-brand {
        text-align: center;
        padding: 50px 20px 30px 20px;
    }

    .hugo-mark {
        display: inline-flex;
        width: 82px;
        height: 82px;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(0, 229, 255, 0.8);
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(0, 229, 255, 0.12),
                rgba(110, 60, 255, 0.12)
            );
        box-shadow:
            0 0 25px rgba(0, 229, 255, 0.18),
            inset 0 0 20px rgba(110, 60, 255, 0.08);
    }

    .hugo-mark span {
        font-size: 30px;
        font-weight: 900;
        color: #ffffff;
        text-shadow:
            2px 0 #00e5ff,
            -2px 0 #7a5cff;
    }

    .hugo-title {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: 14px;
        margin-top: 18px;
        color: white;
        text-shadow:
            3px 0 rgba(0, 229, 255, 0.8),
            -3px 0 rgba(120, 80, 255, 0.8);
    }

    .hugo-subtitle {
        color: #7f9aac;
        font-size: 13px;
        letter-spacing: 5px;
        text-transform: uppercase;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 28px;
        margin-bottom: 25px;
        border: 1px solid rgba(0, 229, 255, 0.12);
        border-radius: 18px;
        background: rgba(5, 13, 22, 0.75);
        backdrop-filter: blur(14px);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 13px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid rgba(0, 229, 255, 0.6);
        background: rgba(0, 229, 255, 0.07);
        color: #00e5ff;
        font-weight: 900;
        font-size: 20px;
    }

    .brand-name {
        font-size: 25px;
        font-weight: 900;
        letter-spacing: 3px;
    }

    .brand-small {
        color: #6f8999;
        font-size: 11px;
        letter-spacing: 2px;
    }

    .live {
        border: 1px solid rgba(0, 255, 170, 0.3);
        color: #5dffbd;
        background: rgba(0, 255, 170, 0.06);
        border-radius: 30px;
        padding: 8px 15px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* ======================================================
       PANELS
       ====================================================== */

    .panel {
        border: 1px solid rgba(105, 145, 170, 0.14);
        background: rgba(5, 13, 22, 0.76);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(15px);
    }

    .panel-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #dff8ff;
        margin-bottom: 14px;
    }

    .panel-subtitle {
        color: #728b9a;
        font-size: 12px;
    }

    /* ======================================================
       VERDICT
       ====================================================== */

    .verdict {
        border-radius: 22px;
        padding: 30px;
        margin: 25px 0;
        text-align: center;
        background:
            linear-gradient(
                135deg,
                rgba(255, 40, 90, 0.13),
                rgba(255, 40, 90, 0.025)
            );
        border: 2px solid rgba(255, 70, 100, 0.55);
        box-shadow:
            0 0 45px rgba(255, 40, 90, 0.10);
    }

    .verdict-label {
        color: #8195a2;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    .verdict-main {
        color: #ff5575;
        font-size: 44px;
        font-weight: 900;
        letter-spacing: 2px;
        margin: 10px 0;
    }

    .verdict-score {
        color: white;
        font-size: 25px;
        font-weight: 800;
    }

    .verdict-note {
        color: #8399a7;
        font-size: 12px;
        margin-top: 10px;
    }

    .verdict-low {
        background:
            linear-gradient(
                135deg,
                rgba(0, 255, 170, 0.10),
                rgba(0, 255, 170, 0.025)
            );
        border-color: rgba(0, 255, 170, 0.45);
    }

    .verdict-low .verdict-main {
        color: #50f5b5;
    }

    .verdict-moderate {
        background:
            linear-gradient(
                135deg,
                rgba(255, 190, 60, 0.10),
                rgba(255, 190, 60, 0.025)
            );
        border-color: rgba(255, 190, 60, 0.45);
    }

    .verdict-moderate .verdict-main {
        color: #ffc84d;
    }

    /* ======================================================
       EVIDENCE CARDS
       ====================================================== */

    .evidence-card {
        background: rgba(7, 17, 28, 0.85);
        border: 1px solid rgba(90, 130, 150, 0.16);
        border-radius: 17px;
        padding: 20px;
        min-height: 130px;
    }

    .evidence-title {
        color: #718a99;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }

    .evidence-value {
        color: white;
        font-size: 27px;
        font-weight: 900;
        margin-top: 8px;
    }

    .evidence-desc {
        color: #758c9a;
        font-size: 11px;
        margin-top: 6px;
    }

    /* ======================================================
       FRAME CARDS
       ====================================================== */

    .frame-card {
        border: 1px solid rgba(255, 70, 100, 0.30);
        border-radius: 15px;
        background: rgba(12, 18, 28, 0.90);
        overflow: hidden;
        margin-bottom: 18px;
    }

    .frame-card-header {
        padding: 13px 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 40, 90, 0.06);
    }

    .frame-id {
        font-size: 13px;
        font-weight: 800;
        color: white;
    }

    .frame-time {
        color: #7f96a4;
        font-size: 11px;
    }

    .fake-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1px;
        background: rgba(255, 50, 80, 0.14);
        color: #ff6580;
        border: 1px solid rgba(255, 50, 80, 0.30);
    }

    .real-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1px;
        background: rgba(0, 255, 170, 0.09);
        color: #5dffbd;
        border: 1px solid rgba(0, 255, 170, 0.25);
    }

    .reason {
        padding: 14px 15px;
        color: #a8bac4;
        font-size: 12px;
        line-height: 1.6;
    }

    /* ======================================================
       TIMELINE
       ====================================================== */

    .timeline {
        position: relative;
        padding: 10px 0;
    }

    .timeline-line {
        height: 5px;
        width: 100%;
        border-radius: 10px;
        background:
            linear-gradient(
                90deg,
                #18e7ff,
                #7666ff,
                #ff4d73
            );
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.15);
    }

    .timeline-label {
        display: flex;
        justify-content: space-between;
        color: #718895;
        font-size: 10px;
        margin-top: 8px;
    }

    /* ======================================================
       ANIMATION NOTICE
       ====================================================== */

    .animation-box {
        border: 1px solid rgba(190, 100, 255, 0.45);
        background: rgba(130, 70, 255, 0.07);
        border-radius: 18px;
        padding: 22px;
        margin: 18px 0;
    }

    .animation-title {
        color: #c68cff;
        font-weight: 900;
        letter-spacing: 2px;
        font-size: 13px;
    }

    .animation-text {
        color: #a795b7;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* ======================================================
       UPLOAD
       ====================================================== */

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(0, 229, 255, 0.35);
        border-radius: 18px;
        background: rgba(0, 229, 255, 0.025);
        padding: 10px;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(0, 229, 255, 0.35);
        background: rgba(0, 229, 255, 0.07);
        color: white;
        font-weight: 800;
        min-height: 45px;
    }

    .stButton > button:hover {
        border-color: #00e5ff;
        background: rgba(0, 229, 255, 0.14);
        color: white;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #425865;
        font-size: 10px;
        letter-spacing: 2px;
        padding: 35px;
        text-transform: uppercase;
    }

    </style>

    <div class="glitch-bg"></div>
    """,
    unsafe_allow_html=True
)

inject_css()


# ============================================================
# HUGO OPENING SCREEN
# ============================================================

if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False


if not st.session_state.intro_seen:

    html(
        """
        <div class="hugo-brand">

            <div class="hugo-mark">
                <span>H</span>
            </div>

            <div class="hugo-title">
                HUGO
            </div>

            <div class="hugo-subtitle">
                GLITCH TECH • FORENSIC INTELLIGENCE
            </div>

        </div>
            """)

    time.sleep(1.5)

    html(
        """
        <div style="
            text-align:center;
            color:#6d8796;
            font-size:12px;
            letter-spacing:3px;
        ">
            INITIALIZING DEEPTRACE FORENSIC ENGINE...
        </div>
        """)

    time.sleep(1.0)

    st.session_state.intro_seen = True
    st.rerun()


# ============================================================
# HEADER
# ============================================================

html(
    """
    <div class="header">

        <div class="brand-left">

            <div class="brand-icon">
                H
            </div>

            <div>
                <div class="brand-name">
                    DEEPTRACE
                </div>

                <div class="brand-small">
                    GLITCH TECH / MULTIMODAL AI FORENSICS
                </div>
            </div>

        </div>

        <div class="live">
            ● FORENSIC ENGINE ONLINE
        </div>

    </div>
    """)


# ============================================================
# INTRODUCTION
# ============================================================

html(
    """
    <div class="panel">

        <div class="panel-title">
            Video Evidence Acquisition
        </div>

        <div class="panel-subtitle">
            Upload a video for frame, facial, temporal, audio and
            audiovisual forensic analysis.
        </div>

    </div>
    """)


uploaded_file = st.file_uploader(
    "Upload video evidence",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm"
    ]
)

with st.expander("Batch analysis and case history"):
    batch_files = st.file_uploader(
        "Choose multiple video files",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
        key="batch_video_uploads",
    )
    batch_interval = st.select_slider(
        "Batch frame interval (seconds)",
        options=[0.5, 1.0, 2.0],
        value=1.0,
        key="batch_frame_interval",
    )
    if st.button(
        "Analyze selected batch",
        disabled=not batch_files,
        key="analyze_video_batch",
    ):
        completed_cases = []
        failures = []
        st.session_state.batch_failures = []
        batch_progress = st.progress(0)
        for index, batch_file in enumerate(batch_files):
            temp_path = None
            try:
                suffix = Path(batch_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(batch_file.getbuffer())
                    temp_path = temp_file.name

                result = analyze_video_file(
                    temp_path,
                    batch_file.name,
                    frame_interval=batch_interval,
                )
                case = save_case(
                    batch_file.name,
                    result["file_hash"],
                    result,
                )
                result["case_summary"] = case
                completed_cases.append(result)
            except Exception as error:
                failures.append({
                    "filename": batch_file.name,
                    "error": str(error),
                })
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            batch_progress.progress((index + 1) / len(batch_files))

        st.session_state.batch_failures = failures
        st.session_state.batch_cases = [
            case["case_summary"] for case in completed_cases
        ]
        if completed_cases:
            store_analysis_result(completed_cases[-1])
            st.success(
                f"Completed {len(completed_cases)} of {len(batch_files)} video analyses. "
                "Showing the most recent completed case below."
            )
            st.rerun()
        if failures:
            st.error(f"{len(failures)} video(s) could not be analyzed.")
            st.dataframe(failures, hide_index=True)

    if st.session_state.get("batch_failures"):
        st.error(
            f'{len(st.session_state.batch_failures)} video(s) failed in the last batch.'
        )
        st.dataframe(st.session_state.batch_failures, hide_index=True)

    case_history = list_cases()
    if case_history:
        st.dataframe(
            [
                {
                    "Analyzed": case["created_at"],
                    "File": case["filename"],
                    "Verdict": case.get("verdict", ""),
                    "Score": f'{case.get("primary_score", 0.0):.1%}',
                    "Faces": case.get("face_count", 0),
                    "SHA-256": case["sha256"],
                }
                for case in case_history
            ],
            hide_index=True,
        )
    else:
        st.caption("No cases have been saved yet.")

with st.expander("Evaluate calibration on labeled predictions"):
    calibration_file = st.file_uploader(
        "CSV with score,label,split columns",
        type=["csv"],
        key="calibration_csv",
    )
    st.caption(
        "Use independent train and held-out rows. Labels: 0 = real, 1 = manipulated; scores must be in [0, 1]."
    )
    if st.button(
        "Evaluate held-out calibration",
        disabled=calibration_file is None,
        key="evaluate_calibration",
    ):
        try:
            calibration = evaluate_calibration_csv(
                io.BytesIO(calibration_file.getvalue())
            )
            save_calibration(calibration)
            metrics = st.columns(6)
            metrics[0].metric("Raw accuracy", f'{calibration["raw_accuracy"]:.1%}')
            metrics[1].metric("Calibrated accuracy", f'{calibration["calibrated_accuracy"]:.1%}')
            metrics[2].metric("Raw Brier", f'{calibration["raw_brier"]:.3f}')
            metrics[3].metric("Calibrated Brier", f'{calibration["calibrated_brier"]:.3f}')
            metrics[4].metric("Expected calibration error", f'{calibration["calibrated_ece"]:.3f}')
            metrics[5].metric("Held-out ROC AUC", f'{calibration["calibrated_auc"]:.3f}')
            st.caption(
                f'Isotonic mapping saved. Trained on {calibration["train_rows"]} rows and evaluated on {calibration["test_rows"]} held-out rows. These metrics describe only the supplied dataset.'
            )
        except Exception as error:
            st.error(f"Calibration evaluation failed: {error}")


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def animation_heuristic(frame_paths, face_results):
    """
    Rule-based animation/CGI appearance heuristic.

    This does NOT prove that animation was AI-generated.
    It looks for:
      - no detected faces
      - high colour saturation
      - strong edge structure
      - repeated visual characteristics
    """

    if not frame_paths:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "No frames available."
        }

    total_faces = sum(
        item.get("face_count", 0)
        for item in face_results
    )

    if total_faces > 0:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "Human faces were detected."
        }

    scores = []

    for item in frame_paths[:20]:

        path = item["path"]

        image = cv2.imread(path)

        if image is None:
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = float(
            np.mean(hsv[:, :, 1]) / 255.0
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_density = float(
            np.mean(edges > 0)
        )

        score = (
            saturation * 0.55 +
            edge_density * 0.45
        )

        scores.append(score)

    if not scores:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "Visual characteristics could not be measured."
        }

    visual_score = float(np.mean(scores))

    if visual_score >= 0.45:
        return {
            "detected": True,
            "confidence": min(
                1.0,
                visual_score + 0.40
            ),
            "reason": (
                "No human faces were detected and the sampled "
                "frames show strong synthetic/illustrated visual "
                "characteristics. This appearance heuristic is not "
                "proof of AI generation."
            )
        }

    return {
        "detected": False,
        "confidence": visual_score,
        "reason": (
            "No faces were detected, but the visual characteristics "
            "were not strong enough to classify the video as animation."
        )
    }


def create_frame_reason(
    fake_probability,
    real_probability,
    temporal_variation
):
    """
    Generates a human-readable forensic explanation
    for a suspicious frame.
    """

    reasons = []

    if fake_probability >= 0.75:
        reasons.append(
            "The facial model produced strong AI/deepfake evidence."
        )

    elif fake_probability >= 0.55:
        reasons.append(
            "The facial model produced moderate AI/deepfake evidence."
        )

    elif fake_probability >= 0.40:
        reasons.append(
            "The frame contains borderline AI/deepfake evidence."
        )

    else:
        reasons.append(
            "The facial model produced stronger real-video evidence."
        )

    if abs(fake_probability - real_probability) < 0.10:
        reasons.append(
            "The model probabilities are close, so this frame should "
            "be treated as uncertain."
        )

    if temporal_variation > 0.10:
        reasons.append(
            "The prediction differs noticeably from nearby frames."
        )

    elif temporal_variation <= 0.05:
        reasons.append(
            "The prediction is relatively stable compared with nearby frames."
        )

    return " ".join(reasons)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file:

    html(
        f"""
        <div class="panel">

            <div class="panel-title">
                Evidence File
            </div>

            <div style="
                color:white;
                font-size:16px;
                font-weight:700;
            ">
                {uploaded_file.name}
            </div>

        </div>
        """,
    )

    frame_interval = st.select_slider(
        "Frame sampling interval (seconds)",
        options=[0.25, 0.5, 1.0, 2.0],
        value=0.5,
        key="frame_sampling_interval",
    )

    analyze_button = st.button(
        "◈ START FORENSIC ANALYSIS",
        width="stretch"
    )

    if analyze_button:

        try:

            scan_slot = st.empty()
            log = TerminalLog()
            scan_image = None

            def step(label, pct):
                log.log(label)
                scan_panel(scan_slot, label, pct, scan_image)

            step("Acquiring video evidence", 0)

            # ------------------------------------------------
            # SAVE TEMP VIDEO
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_video = temp_file.name

            file_hash = sha256_file(temp_video)
            provenance = probe_video_metadata(temp_video)
            case_id = file_hash[:16]
            step("File provenance captured", 10)

            # ------------------------------------------------
            # SCENE DETECTION
            # ------------------------------------------------

            step("Detecting scenes", 10)

            scenes = detect_scenes(temp_video)
            step("Scene detection complete", 20)

            # ------------------------------------------------
            # FRAME EXTRACTION
            # ------------------------------------------------

            step("Extracting forensic frames", 20)

            frame_folder = os.path.join(
                "frames",
                case_id,
            )

            os.makedirs(
                frame_folder,
                exist_ok=True
            )

            frames = extract_frames(
                temp_video,
                frame_folder,
                interval=frame_interval,
            )

            if not frames:
                raise RuntimeError(
                    "No frames could be extracted from the video."
                )

            scan_image = frames[len(frames) // 2]["path"]
            step(f"Extracted {len(frames)} sampled frames", 35)

            # ------------------------------------------------
            # FACE / VIDEO ANALYSIS
            # ------------------------------------------------

            step("Running facial AI/deepfake analysis", 35)

            video_results = analyze_frames(
                frames,
                face_folder=os.path.join(
                    "faces",
                    case_id,
                ),
            )

            step("Frame-level video analysis complete", 55)

            # ------------------------------------------------
            # TEMPORAL ANALYSIS
            # ------------------------------------------------

            step("Analyzing temporal consistency", 55)

            try:

                temporal_result = (
                    analyze_temporal_consistency(
                        video_results
                    )
                )

                temporal_consistency = (
                    temporal_result[
                        "temporal_consistency"
                    ]
                )

                video_score = (
                    temporal_result[
                        "mean_fake_probability"
                    ]
                )

            except ValueError:

                temporal_result = {
                    "frames_analyzed": len(video_results),
                    "faces_analyzed": 0,
                    "mean_fake_probability": 0.0,
                    "average_deviation": 0.0,
                    "temporal_consistency": 0.0,
                    "frame_scores": []
                }

                temporal_consistency = 0.0
                video_score = 0.0

            step("Temporal consistency analysis complete", 65)

            # ------------------------------------------------
            # AUDIO EXTRACTION
            # ------------------------------------------------

            step("Extracting audio evidence", 65)

            audio_path = os.path.join(
                "audio",
                "deeptrace_current.wav"
            )

            extract_audio(
                temp_video,
                audio_path
            )

            step("Audio extraction complete", 72)

            # ------------------------------------------------
            # AUDIO MODEL
            # ------------------------------------------------

            step("Running AASIST-L audio analysis", 72)

            audio_model = AudioDeepfakeModel()
            audio_model.load_model()

            audio_windows = analyze_audio_windows(
                audio_path,
                audio_model
            )

            audio_evidence = score_audio_evidence(
                audio_windows
            )

            audio_score = (
                audio_evidence[
                    "evidence_score"
                ]
            )

            step("Audio window analysis complete", 82)

            # ------------------------------------------------
            # AUDIO INFO
            # ------------------------------------------------

            audio_info = analyze_audio(
                audio_path
            )

            # ------------------------------------------------
            # VIDEO DURATION
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                temp_video
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            frame_count = cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )

            if fps > 0:
                video_duration = (
                    frame_count / fps
                )
            else:
                video_duration = 0.0

            cap.release()

            # ------------------------------------------------
            # A/V SYNC
            # ------------------------------------------------

            sync_result = analyze_av_sync(
                video_duration,
                audio_info["duration"]
            )

            step("Analyzing lip synchronization", 82)
            try:
                lip_sync = analyze_lip_sync(temp_video, audio_path)
            except Exception as error:
                lip_sync = lip_sync_unavailable(str(error))

            step("Lip-sync analysis complete", 88)

            # ------------------------------------------------
            # MULTIMODAL SCORE
            # ------------------------------------------------

            multimodal_result = (
                calculate_multimodal_score(
                    video_score,
                    audio_score,
                    temporal_consistency,
                    sync_result["sync_status"]
                )
            )
            step("Combining multimodal evidence", 90)

            # ------------------------------------------------
            # ANIMATION ANALYSIS
            # ------------------------------------------------

            animation_result = animation_heuristic(
                frames,
                video_results
            )

            step("Visual animation screening complete", 95)

            # ------------------------------------------------
            # FORENSIC REPORT
            # ------------------------------------------------

            report = generate_forensic_report(
                filename=uploaded_file.name,
                multimodal_result=multimodal_result,
                video_result={
                    "mean_fake_probability":
                        video_score,
                    "frames_analyzed":
                        temporal_result[
                            "frames_analyzed"
                        ],
                    "faces_analyzed":
                        temporal_result[
                            "faces_analyzed"
                        ]
                },
                audio_result={
                    "average_spoof_probability":
                        audio_windows[
                            "average_spoof_probability"
                        ],
                    "maximum_spoof_probability":
                        audio_windows[
                            "maximum_spoof_probability"
                        ],
                    "spoof_window_percentage":
                        audio_windows[
                            "spoof_window_percentage"
                        ]
                },
                temporal_result={
                    "temporal_consistency":
                        temporal_consistency,
                    "average_deviation":
                        temporal_result[
                            "average_deviation"
                        ]
                },
                sync_result=sync_result
            )

            step("Forensic report generated", 99)

            # =================================================
            # SAVE RESULTS
            # =================================================

            st.session_state.analysis_complete = True

            st.session_state.video_results = (
                video_results
            )

            st.session_state.frames = frames

            st.session_state.scenes = scenes

            st.session_state.temporal_result = (
                temporal_result
            )

            st.session_state.audio_windows = (
                audio_windows
            )

            st.session_state.audio_evidence = (
                audio_evidence
            )

            st.session_state.sync_result = (
                sync_result
            )

            st.session_state.lip_sync = lip_sync

            st.session_state.multimodal_result = (
                multimodal_result
            )

            st.session_state.animation_result = (
                animation_result
            )

            st.session_state.report = report

            st.session_state.filename = (
                uploaded_file.name
            )

            st.session_state.file_hash = file_hash
            st.session_state.provenance = provenance
            st.session_state.frame_interval = frame_interval
            st.session_state.syncnet = {
                "available": False,
                "status": "No compatible SyncNet model code or pretrained checkpoint is configured.",
            }
            save_case(
                uploaded_file.name,
                file_hash,
                {
                    "multimodal_result": multimodal_result,
                    "video_results": video_results,
                    "audio_windows": audio_windows,
                },
            )

            log.finish("Forensic analysis completed")
            time.sleep(1.2)
            st.rerun()

        except Exception as error:

            if "log" in locals():
                log.error(str(error))

            st.error(
                "The analysis pipeline encountered a problem."
            )

            st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.get(
    "analysis_complete",
    False
):

    video_results = (
        st.session_state.video_results
    )

    frames = (
        st.session_state.frames
    )

    scenes = (
        st.session_state.scenes
    )

    temporal_result = (
        st.session_state.temporal_result
    )

    audio_windows = (
        st.session_state.audio_windows
    )

    audio_evidence = (
        st.session_state.audio_evidence
    )

    sync_result = (
        st.session_state.sync_result
    )

    multimodal_result = (
        st.session_state.multimodal_result
    )

    animation_result = (
        st.session_state.animation_result
    )

    report = (
        st.session_state.report
    )

    file_hash = st.session_state.get(
        "file_hash",
        "Not recorded for this earlier analysis",
    )
    provenance = st.session_state.get(
        "provenance",
        {
            "available": False,
            "error": "Metadata was not recorded for this earlier analysis.",
        },
    )
    syncnet = st.session_state.get(
        "syncnet",
        {
            "available": False,
            "status": "SyncNet was not run for this analysis.",
        },
    )
    lip_sync = st.session_state.get("lip_sync")

    video_score = temporal_result[
        "mean_fake_probability"
    ]
    audio_score = audio_evidence[
        "evidence_score"
    ]
    temporal_consistency = temporal_result[
        "temporal_consistency"
    ]

    streamlit_module = st
    tab_overview, tab_video, tab_audio, tab_timeline, tab_report = st.tabs(
        ["Overview", "Video", "Audio", "Timeline", "Report"]
    )
    st = _TabRenderer(tab_overview, streamlit_module)

    # ========================================================
    # MAIN VERDICT
    # ========================================================

    primary_score = (
        multimodal_result[
            "primary_score"
        ]
    )
    active_calibration = load_calibration()
    calibrated_primary_score = apply_calibration(
        primary_score,
        active_calibration,
    )

    verdict = (
        multimodal_result[
            "verdict"
        ]
    )

    g1, g2 = st.columns([1, 1.4], gap="large")
    with g1:
        render_gauge(
            primary_score,
            verdict,
            "PRIMARY EVIDENCE SCORE",
            "Raw evidence score; not a calibrated probability.",
        )
        if calibrated_primary_score is not None:
            st.caption(
                f"Calibrated score: {calibrated_primary_score:.2%}. "
                "Valid only for data similar to the calibration set."
            )
    with g2:
        evidence_columns = st.columns(2)
        evidence_items = (
            ("Video Evidence", f"{video_score:.1%}", "Mean facial AI evidence"),
            ("Audio Evidence", f"{audio_score:.1%}", "AASIST-L evidence score"),
            ("Temporal Stability", f"{temporal_consistency:.1%}", "Prediction stability across frames"),
            ("A/V Synchronization", sync_result["sync_status"], "Audio/video duration comparison"),
        )
        for column, item in zip(
            (evidence_columns[0], evidence_columns[1], evidence_columns[0], evidence_columns[1]),
            evidence_items,
        ):
            with column:
                st.metric(item[0], item[1], border=True, help=item[2])

    with st.expander("File integrity and media metadata"):
        st.caption("SHA-256 file hash")
        st.code(file_hash, language=None)
        st.caption(
            "Container metadata can support an investigation, but does not by itself prove editing or authenticity."
        )
        if provenance.get("available"):
            metadata_rows = [
                {
                    "Property": label,
                    "Value": str(provenance.get(key) or "Not present"),
                }
                for label, key in (
                    ("Container", "format"),
                    ("Duration (seconds)", "duration_seconds"),
                    ("Video codec", "video_codec"),
                    ("Resolution", "width"),
                    ("Height", "height"),
                    ("Encoder tag", "encoder"),
                    ("Creation time", "creation_time"),
                    ("Audio codec", "audio_codec"),
                    ("Audio sample rate", "audio_sample_rate"),
                    ("Audio channels", "audio_channels"),
                )
            ]
            st.dataframe(metadata_rows, hide_index=True)
        else:
            st.warning(provenance.get("error", "Metadata is unavailable."))

    # ========================================================
    # ANIMATION / CGI
    # ========================================================

    if animation_result["detected"]:

        html(
            f"""
            <div class="animation-box">

                <div class="animation-title">
                    ◈ ANIMATION / CGI CHARACTERISTICS DETECTED
                </div>

                <div class="animation-text">
                    Rule-based appearance score:
                    <strong>
                        {animation_result["confidence"] * 100:.2f}%
                    </strong>
                    <br><br>
                    {animation_result["reason"]}
                </div>

            </div>
            """)

    # ========================================================
    # FRAME-LEVEL ANALYSIS
    # ========================================================

    st = _TabRenderer(tab_video, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Frame-Level AI Evidence
            </div>

            <div class="panel-subtitle">
                Each sampled frame is inspected independently.
                Suspicious frames are displayed below with the
                reason for their classification.
            </div>

        </div>
        """)

    # --------------------------------------------------------
    # BUILD FRAME SCORES
    # --------------------------------------------------------

    frame_analysis = []

    all_fake_scores = []

    for result in video_results:

        fake_scores = []

        for face in result.get(
            "faces",
            []
        ):

            if "fake_probability" in face:

                fake_scores.append(
                    face[
                        "fake_probability"
                    ]
                )

        if fake_scores:

            frame_fake = float(
                np.mean(fake_scores)
            )

            frame_real = 1.0 - frame_fake

            all_fake_scores.append(
                frame_fake
            )

            frame_analysis.append(
                {
                    "frame":
                        result.get(
                            "frame"
                        ),
                    "faces": result.get("faces", []),
                    "timestamp":
                        result.get(
                            "timestamp"
                        ),
                    "face_count":
                        result.get(
                            "face_count",
                            0
                        ),
                    "fake_probability":
                        frame_fake,
                    "real_probability":
                        frame_real
                }
            )

        else:

            frame_analysis.append(
                {
                    "frame":
                        result.get(
                            "frame"
                        ),
                    "faces": result.get("faces", []),
                    "timestamp":
                        result.get(
                            "timestamp"
                        ),
                    "face_count":
                        result.get(
                            "face_count",
                            0
                        ),
                    "fake_probability":
                        None,
                    "real_probability":
                        None
                }
            )

    # --------------------------------------------------------
    # ADD LOCAL VARIATION
    # --------------------------------------------------------

    for index, item in enumerate(
        frame_analysis
    ):

        if item[
            "fake_probability"
        ] is None:

            item[
                "variation"
            ] = 0.0

            continue

        nearby = []

        if index > 0:

            previous = frame_analysis[
                index - 1
            ]

            if previous[
                "fake_probability"
            ] is not None:

                nearby.append(
                    previous[
                        "fake_probability"
                    ]
                )

        if index < len(
            frame_analysis
        ) - 1:

            following = frame_analysis[
                index + 1
            ]

            if following[
                "fake_probability"
            ] is not None:

                nearby.append(
                    following[
                        "fake_probability"
                    ]
                )

        if nearby:

            item[
                "variation"
            ] = float(
                np.mean(
                    [
                        abs(
                            item[
                                "fake_probability"
                            ] - value
                        )
                        for value in nearby
                    ]
                )
            )

        else:

            item[
                "variation"
            ] = 0.0

    selected_frame = frame_viewer(
        frame_analysis,
        reason_fn=create_frame_reason,
    )

    if selected_frame:
        with st.expander("Compression / ELA diagnostic"):
            ela_result = error_level_analysis(selected_frame["frame"])
            st.image(
                ela_result["visualization"],
                caption=f'ELA visualization · JPEG quality {ela_result["quality"]}',
                alt="Error level analysis visualization for the selected frame",
                width="stretch",
            )
            ela_metrics = st.columns(2)
            ela_metrics[0].metric(
                "Mean recompression difference",
                f'{ela_result["mean_difference"]:.2f}',
            )
            ela_metrics[1].metric(
                "95th percentile difference",
                f'{ela_result["p95_difference"]:.2f}',
            )
            st.caption(
                "ELA highlights differences after JPEG recompression. It is sensitive to image content and compression history; it is not proof that a region was edited."
            )

    # --------------------------------------------------------
    # SUSPICIOUS FRAMES
    # --------------------------------------------------------

    suspicious_frames = [
        item
        for item in frame_analysis
        if item[
            "fake_probability"
        ] is not None
        and item[
            "fake_probability"
        ] >= 0.50
    ]

    if suspicious_frames:

        html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    ⚠ Suspicious Frame Locations
                </div>

                <div class="panel-subtitle">
                    {len(suspicious_frames)}
                    sampled frame(s) contain moderate or stronger
                    facial AI/deepfake evidence.
                </div>

            </div>
            """)

        # Show strongest frames first

        suspicious_frames = sorted(
            suspicious_frames,
            key=lambda x:
                x["fake_probability"],
            reverse=True
        )

        # Limit visual gallery to strongest 12

        display_frames = (
            suspicious_frames[:12]
        )

        cols = st.columns(3)

        for index, item in enumerate(
            display_frames
        ):

            with cols[
                index % 3
            ]:

                path = item["frame"]

                if os.path.exists(
                    path
                ):

                    st.image(
                        path,
                        width="stretch"
                    )

                fake = (
                    item[
                        "fake_probability"
                    ]
                )

                reason = create_frame_reason(
                    fake,
                    item[
                        "real_probability"
                    ],
                    item[
                        "variation"
                    ]
                )

                html(
                    f"""
                    <div class="frame-card">

                        <div class="frame-card-header">

                            <div>
                                <div class="frame-id">
                                    FRAME
                                    {index + 1}
                                </div>

                                <div class="frame-time">
                                    Timestamp:
                                    {item["timestamp"]:.2f}s
                                </div>
                            </div>

                            <span class="fake-badge">
                                AI SUSPECTED
                            </span>

                        </div>

                        <div class="reason">

                            <strong>
                                AI evidence:
                                {fake * 100:.2f}%
                            </strong>

                            <br><br>

                            {reason}

                        </div>

                    </div>
                    """)

    else:

        st.success(
            "No sampled frame reached the moderate AI/deepfake "
            "evidence threshold."
        )

    # ========================================================
    # NO-FACE DETECTION
    # ========================================================

    no_face_frames = [
        item
        for item in frame_analysis
        if item[
            "face_count"
        ] == 0
    ]
    frames_with_faces = len(frame_analysis) - len(no_face_frames)

    if no_face_frames:
        if frames_with_faces:
            face_status = (
                f"FACES DETECTED IN {frames_with_faces} OF "
                f"{len(frame_analysis)} SAMPLED FRAMES"
            )
            status_color = "#5dffbd"
            face_detail = (
                f"No face boxes were detected in {len(no_face_frames)} "
                "sampled frames."
            )
        else:
            face_status = "NO FACES DETECTED IN SAMPLED FRAMES"
            status_color = "#ffc84d"
            face_detail = (
                f"No face boxes were detected in any of the "
                f"{len(frame_analysis)} sampled frames."
            )

        html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Face Detection Status
                </div>

                <div style="
                    color:{status_color};
                    font-size:18px;
                    font-weight:900;
                ">
                    {face_status}
                </div>

                <div class="panel-subtitle"
                     style="margin-top:10px;">
                    {face_detail} Face-based analysis is unavailable
                    for those frames.
                </div>

            </div>
            """)
        st.caption(
            "Face detection can miss small, profile, occluded, or stylized faces. "
            "Review the sampled frame viewer if a visible face was missed."
        )

    # ========================================================
    # ALL FRAME TABLE
    # ========================================================

    st = _TabRenderer(tab_timeline, streamlit_module)

    html("""
    <div class="panel">
    <div class="panel-title">Evidence Timeline</div>
    <div class="panel-subtitle">Video, audio and lip-sync evidence on one time axis.
    Red bands mark where evidence is suspicious.</div>
    </div>
    """)

    st.plotly_chart(
        build_timeline_figure(
            frame_analysis,
            audio_windows,
            st.session_state.get("lip_sync"),
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Complete Frame Timeline
            </div>

        </div>
        """)

    timeline_data = []

    for index, item in enumerate(
        frame_analysis
    ):

        if item[
            "fake_probability"
        ] is None:

            prediction = (
                "NO FACES DETECTED"
            )

            ai_score = "N/A"

        else:

            ai_score = (
                f'{item["fake_probability"] * 100:.2f}%'
            )

            if item[
                "fake_probability"
            ] >= 0.50:

                prediction = (
                    "AI SUSPECTED"
                )

            else:

                prediction = (
                    "REAL / LOW EVIDENCE"
                )

        timeline_data.append(
            {
                "Frame":
                    index + 1,
                "Time (s)":
                    round(
                        item["timestamp"],
                        2
                    ),
                "Faces":
                    item["face_count"],
                "AI Evidence":
                    ai_score,
                "Status":
                    prediction
            }
        )

    st.dataframe(
        timeline_data,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # SCENE ANALYSIS
    # ========================================================

    st = _TabRenderer(tab_video, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Scene Detection
            </div>

            <div class="panel-subtitle">
                Detected scene boundaries from the uploaded video.
            </div>

        </div>
        """)

    if scenes:

        scene_data = []

        for scene in scenes:

            scene_data.append(
                {
                    "Scene":
                        scene["scene"],
                    "Start":
                        f'{scene["start"]:.2f}s',
                    "End":
                        f'{scene["end"]:.2f}s',
                    "Duration":
                        f'{scene["end"] - scene["start"]:.2f}s'
                }
            )

        st.dataframe(
            scene_data,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No scene changes were detected."
        )

    # ========================================================
    # AUDIO ANALYSIS
    # ========================================================

    st = _TabRenderer(tab_audio, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Audio Forensics
            </div>

        </div>
        """)

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Average Spoof Probability",
            f'{audio_windows["average_spoof_probability"] * 100:.2f}%'
        )

    with a2:

        st.metric(
            "Maximum Spoof Probability",
            f'{audio_windows["maximum_spoof_probability"] * 100:.2f}%'
        )

    with a3:

        st.metric(
            "Suspicious Audio Windows",
            f'{audio_windows["spoof_window_percentage"]:.2f}%'
        )

    audio_table = []

    for window in audio_windows[
        "windows"
    ]:

        audio_table.append(
            {
                "Window":
                    window["window"],
                "Start":
                    f'{window["start_time"]:.2f}s',
                "End":
                    f'{window["end_time"]:.2f}s',
                "Spoof Probability":
                    f'{window["spoof_probability"] * 100:.2f}%',
                "Prediction":
                    window["prediction"]
            }
        )

    st.dataframe(
        audio_table,
        width="stretch",
        hide_index=True
    )

    st = _TabRenderer(tab_timeline, streamlit_module)

    timeline_rows = [
        {
            "time": item["timestamp"],
            "evidence": item["fake_probability"],
            "source": "Video",
        }
        for item in frame_analysis
        if item["fake_probability"] is not None
    ]
    timeline_rows.extend(
        {
            "time": window["start_time"],
            "evidence": window["spoof_probability"],
            "source": "Audio",
        }
        for window in audio_windows["windows"]
    )

    if timeline_rows:
        timeline_frame = pd.DataFrame(timeline_rows)
        timeline_base = alt.Chart(timeline_frame).encode(
            x=alt.X("time:Q", title="Time (seconds)"),
            y=alt.Y(
                "evidence:Q",
                title="Model evidence score",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("source:N", title="Evidence source"),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("time:Q", title="Time", format=".2f"),
                alt.Tooltip("evidence:Q", title="Score", format=".1%"),
            ],
        )
        evidence_lines = timeline_base.mark_line(point=True)
        suspicious_points = (
            timeline_base
            .transform_filter(alt.datum.evidence >= 0.5)
            .mark_point(shape="triangle", size=100, color="#ff5575")
        )
        st.altair_chart(
            (evidence_lines + suspicious_points).interactive(),
            alt="Video face-model and audio spoof-model evidence over sampled time",
        )
        st.caption(
            "Triangles mark scores at or above the app's 50% review threshold. Video frames and audio windows use different sampling intervals. A/V synchronization is a global duration comparison, not time-resolved lip-sync analysis."
        )

    # ========================================================
    # TEMPORAL ANALYSIS
    # ========================================================

    st = _TabRenderer(tab_video, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Temporal Analysis
            </div>

        </div>
        """)

    t1, t2, t3 = st.columns(3)

    with t1:

        st.metric(
            "Frames Analyzed",
            temporal_result[
                "frames_analyzed"
            ]
        )

    with t2:

        st.metric(
            "Mean AI Evidence",
            f'{temporal_result["mean_fake_probability"] * 100:.2f}%'
        )

    with t3:

        st.metric(
            "Temporal Stability",
            f'{temporal_result["temporal_consistency"] * 100:.2f}%'
        )

    # ========================================================
    # A/V SYNC
    # ========================================================

    st = _TabRenderer(tab_audio, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Audio / Video Synchronization
            </div>

        </div>
        """)

    s1, s2, s3 = st.columns(3)

    with s1:

        st.metric(
            "Video Duration",
            f'{sync_result["video_duration"]:.2f}s'
        )

    with s2:

        st.metric(
            "Audio Duration",
            f'{sync_result["audio_duration"]:.2f}s'
        )

    with s3:

        st.metric(
            "Difference",
            f'{sync_result["duration_difference"]:.2f}s'
        )

    st.info(
        f'A/V status: {sync_result["sync_status"]}'
    )
    st.warning(
        "SyncNet is not configured. The lip-sync result below uses "
        "MediaPipe mouth landmarks and an audio loudness envelope. "
        + syncnet.get("status", "A compatible SyncNet model is unavailable.")
    )

    if lip_sync:
        html("""
        <div class="panel">
        <div class="panel-title">Lip-Sync Analysis</div>
        </div>
        """)

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Status", lip_sync["status"])
        l2.metric("Audio Delay", f'{lip_sync["audio_delay_ms"]:+.0f} ms')
        l3.metric("Correlation", f'{lip_sync["peak_correlation"]:.2f}')
        l4.metric("Face Coverage", f'{lip_sync["face_coverage"] * 100:.0f}%')

        st.info(lip_sync["reason"])

        if lip_sync["mouth_trace"]:
            st.line_chart(
                pd.DataFrame({
                    "Mouth movement": lip_sync["mouth_trace"],
                    "Audio loudness": lip_sync["audio_trace"],
                }),
                alt="Relative mouth motion and audio loudness traces across the video",
            )

        if lip_sync["windows"]:
            st.dataframe(
                lip_sync["windows"],
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st = _TabRenderer(tab_report, streamlit_module)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Export Forensic Results
            </div>

            <div class="panel-subtitle">
                Download the complete forensic report or the
                machine-readable analysis data.
            </div>

        </div>
        """)

    export_data = {
        "filename":
            st.session_state.filename,

        "multimodal_result":
            multimodal_result,

        "temporal_result":
            temporal_result,

        "audio_evidence":
            audio_evidence,

        "audio_windows":
            audio_windows,

        "av_sync":
            sync_result,

        "syncnet":
            syncnet,

        "lip_sync":
            lip_sync,

        "sha256":
            file_hash,

        "raw_primary_score":
            primary_score,

        "calibrated_primary_score":
            calibrated_primary_score,

        "calibration_metrics":
            active_calibration,

        "media_provenance":
            provenance,

        "animation_analysis":
            animation_result,

        "frame_analysis":
            frame_analysis,

        "scenes":
            scenes
    }

    json_results = json.dumps(
        export_data,
        indent=4,
        default=str
    )

    d1, d2 = st.columns(2)

    with d1:

        st.download_button(
            label="⬇ DOWNLOAD FORENSIC REPORT",
            data=report,
            file_name=(
                "DeepTrace_Forensic_Report.txt"
            ),
            mime="text/plain",
            width="stretch"
        )

    with d2:

        st.download_button(
            label="⬇ DOWNLOAD RAW ANALYSIS JSON",
            data=json_results,
            file_name=(
                "DeepTrace_Analysis.json"
            ),
            mime="application/json",
            width="stretch"
        )

    # ========================================================
    # REPORT PREVIEW
    # ========================================================

    with st.expander(
        "VIEW COMPLETE FORENSIC REPORT"
    ):

        st.code(
            report,
            language="text"
        )

    st = streamlit_module

# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="footer">
        HUGO × GLITCH TECH
        <br>
        DEEPTRACE MULTIMODAL FORENSIC INTELLIGENCE
    </div>
    """)
