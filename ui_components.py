import base64
from pathlib import Path

import cv2
import streamlit as st


_COMPONENT_CSS = """
<style>
.scan-card {
    max-width: 760px;
    margin: 16px auto;
    overflow: hidden;
    border: 1px solid rgba(67, 244, 218, .45);
    background: rgba(5, 15, 22, .92);
    box-shadow: 0 0 28px rgba(0, 229, 255, .12);
}
.scan-brand {
    padding: 12px 16px;
    color: #eafaff;
    font: 900 15px/1.2 monospace;
    letter-spacing: 5px;
    text-shadow: 2px 0 #00e5ff, -2px 0 #ff426a;
    animation: hugo-flicker 1.6s steps(2, end) infinite;
}
.scan-image-wrap { position: relative; background: #07131c; }
.scan-image-wrap img { display: block; width: 100%; max-height: 340px; object-fit: contain; }
.scan-line {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: #43f4da;
    box-shadow: 0 0 12px 3px rgba(67, 244, 218, .65);
    animation: scan-sweep 2.2s linear infinite;
}
.scan-caption { padding: 8px 12px; color: #8ceee0; font: 11px monospace; }
.score-gauge {
    --score: 0%;
    --gauge-color: #50f5b5;
    width: min(210px, 100%);
    aspect-ratio: 1;
    margin: 8px auto 20px;
    border-radius: 50%;
    padding: 12px;
    background: conic-gradient(from -90deg, var(--gauge-color) var(--score), rgba(118,145,162,.18) var(--score));
    box-shadow: 0 0 28px color-mix(in srgb, var(--gauge-color) 22%, transparent);
}
.score-gauge-inner {
    display: flex;
    width: 100%;
    height: 100%;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(190,220,235,.14);
    border-radius: 50%;
    background: #08131c;
    text-align: center;
}
.score-gauge-value { color: #f4fbff; font-size: 34px; font-weight: 800; line-height: 1; }
.score-gauge-title { margin-top: 8px; color: #a4bbc8; font-size: 10px; letter-spacing: 1.3px; }
.score-gauge-verdict { margin-top: 6px; color: var(--gauge-color); font-size: 11px; font-weight: 800; }
@keyframes scan-sweep { to { top: 100%; } }
@keyframes hugo-flicker { 0%, 89%, 92%, 100% { opacity: 1; } 90%, 91% { opacity: .55; } }
</style>
"""


def inject_css():
    st.markdown(_COMPONENT_CSS, unsafe_allow_html=True)


class TerminalLog:
    def __init__(self):
        self._placeholder = st.empty()
        self._lines = []

    def _render(self):
        self._placeholder.code("\n".join(self._lines), language="bash")

    def log(self, message):
        self._lines.append(f"[{time_stamp()}] > {message}")
        self._render()

    def finish(self, message):
        self._lines.append(f"[{time_stamp()}] OK  {message}")
        self._render()

    def error(self, message):
        self._lines.append(f"[{time_stamp()}] ERR {message}")
        self._render()


def time_stamp():
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def scan_panel(slot, label, pct, image_path=None):
    image_markup = ""
    path = Path(image_path) if image_path else None
    if path and path.is_file():
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        image_markup = (
            '<div class="scan-image-wrap">'
            f'<img src="data:image/jpeg;base64,{image_data}" '
            'alt="Video frame under forensic scan" />'
            '<div class="scan-line"></div>'
            '</div>'
        )

    with slot.container():
        st.markdown(
            f"""
            <div class="scan-card">
                <div class="scan-brand">HUGO · FORENSIC SCAN</div>
                {image_markup}
                <div class="scan-caption">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(max(0, min(100, int(pct))))


def render_gauge(score, verdict, title, note):
    score_percent = max(0.0, min(1.0, float(score))) * 100
    if score_percent >= 75:
        color = "#ff5575"
    elif score_percent >= 55:
        color = "#ffc84d"
    else:
        color = "#50f5b5"

    st.markdown(
        f"""
        <div class="score-gauge" style="--score:{score_percent:.2f}%;--gauge-color:{color};">
            <div class="score-gauge-inner">
                <div class="score-gauge-value">{score_percent:.1f}%</div>
                <div class="score-gauge-title">{title}</div>
                <div class="score-gauge-verdict">{verdict}</div>
            </div>
        </div>
        <div style="max-width:300px;margin:-8px auto 16px;text-align:center;color:#8399a7;font-size:12px;">
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def frame_viewer(frame_analysis, reason_fn):
    if not frame_analysis:
        st.info("No sampled frames are available to display.")
        return None

    if len(frame_analysis) > 1:
        selected_number = st.slider(
            "Sampled frame",
            min_value=1,
            max_value=len(frame_analysis),
            value=1,
            key="sampled_frame_viewer",
        )
    else:
        selected_number = 1
        st.caption("Sampled frame 1 of 1")

    selected = frame_analysis[selected_number - 1]
    image_path = selected.get("frame")
    original = cv2.imread(str(image_path)) if image_path else None
    if original is None:
        st.warning("The selected sampled frame could not be loaded.")
        return selected

    annotated = original.copy()
    for index, face in enumerate(selected.get("faces", []), 1):
        x = int(face.get("x", 0))
        y = int(face.get("y", 0))
        width = int(face.get("width", 0))
        height = int(face.get("height", 0))
        fake_probability = face.get("fake_probability")
        suspicious = fake_probability is not None and fake_probability >= 0.5
        color = (40, 50, 240) if suspicious else (70, 210, 130)
        label = (
            f"AI evidence {fake_probability * 100:.0f}%"
            if fake_probability is not None
            else f"Face {face.get('face', index)}"
        )
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            annotated,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    original_column, annotated_column = st.columns(2)
    with original_column:
        st.image(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            caption=f"Original · {selected.get('timestamp', 0):.2f}s",
            alt="Original sampled video frame",
            width="stretch",
        )
    with annotated_column:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Detected faces · {selected.get('face_count', 0)}",
            alt="Sampled frame with detected faces and model scores",
            width="stretch",
        )

    score = selected.get("fake_probability")
    if score is not None:
        reason = reason_fn(
            score,
            selected.get("real_probability", 1.0 - score),
            selected.get("variation", 0.0),
        )
        st.caption(f"Frame evidence: {score:.1%}. {reason}")

    return selected
