from collections.abc import Mapping, Sequence

import plotly.graph_objects as go


SUSPICIOUS_THRESHOLD = 0.5


def _as_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _windows(value):
    if isinstance(value, Mapping):
        return value.get("windows", [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def build_timeline_figure(frame_analysis, audio_windows, lip_sync):
    """Build a shared-time-axis view of video, audio, and lip-sync evidence."""
    tracks = ("Video", "Audio", "Lip-sync")
    colors = {
        "suspicious": "#e5484d",
        "video": "#42c8b5",
        "audio": "#f2b84b",
        "lip_sync": "#7d9cff",
    }
    figure = go.Figure()
    max_time = 0.0

    frames = [
        item for item in (frame_analysis or [])
        if isinstance(item, Mapping)
        and _as_float(item.get("timestamp")) is not None
    ]
    frame_times = [_as_float(item.get("timestamp")) for item in frames]

    for index, (item, timestamp) in enumerate(zip(frames, frame_times)):
        previous_time = frame_times[index - 1] if index else None
        next_time = frame_times[index + 1] if index + 1 < len(frame_times) else None
        left = (
            (previous_time + timestamp) / 2
            if previous_time is not None
            else max(0.0, timestamp - 0.25)
        )
        right = (
            (timestamp + next_time) / 2
            if next_time is not None
            else timestamp + 0.25
        )
        score = _as_float(item.get("fake_probability"))
        if score is None:
            continue
        suspicious = score >= SUSPICIOUS_THRESHOLD
        figure.add_trace(
            go.Bar(
                x=[max(0.05, right - left)],
                base=[left],
                y=["Video"],
                orientation="h",
                width=0.52,
                marker_color=colors["suspicious" if suspicious else "video"],
                customdata=[[timestamp, score]],
                hovertemplate=(
                    "Video frame<br>Time: %{customdata[0]:.2f}s"
                    "<br>AI evidence: %{customdata[1]:.1%}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        max_time = max(max_time, timestamp)

    for window in _windows(audio_windows):
        if not isinstance(window, Mapping):
            continue
        start = _as_float(window.get("start_time"))
        end = _as_float(window.get("end_time"))
        score = _as_float(window.get("spoof_probability"))
        if start is None or end is None or score is None or end <= start:
            continue
        suspicious = score >= SUSPICIOUS_THRESHOLD
        figure.add_trace(
            go.Bar(
                x=[end - start],
                base=[start],
                y=["Audio"],
                orientation="h",
                width=0.52,
                marker_color=colors["suspicious" if suspicious else "audio"],
                customdata=[[start, end, score]],
                hovertemplate=(
                    "Audio window<br>%{customdata[0]:.2f}–%{customdata[1]:.2f}s"
                    "<br>Spoof evidence: %{customdata[2]:.1%}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        max_time = max(max_time, end)

    if isinstance(lip_sync, Mapping):
        for window in _windows(lip_sync):
            if not isinstance(window, Mapping):
                continue
            start = _as_float(window.get("start_time"))
            end = _as_float(window.get("end_time"))
            if start is None or end is None or end <= start:
                continue
            status = str(window.get("status", "UNKNOWN"))
            suspicious = status in {"OFFSET DETECTED", "NO CORRELATION"}
            figure.add_trace(
                go.Bar(
                    x=[end - start],
                    base=[start],
                    y=["Lip-sync"],
                    orientation="h",
                    width=0.52,
                    marker_color=colors["suspicious" if suspicious else "lip_sync"],
                    customdata=[[start, end, status]],
                    hovertemplate=(
                        "Lip-sync window<br>%{customdata[0]:.2f}–"
                        "%{customdata[1]:.2f}s<br>Status: "
                        "%{customdata[2]}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            max_time = max(max_time, end)

    figure.update_layout(
        height=190,
        margin={"l": 12, "r": 16, "t": 12, "b": 36},
        barmode="overlay",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce4ed", "size": 12},
        xaxis={
            "title": "Time (seconds)",
            "range": [0, max(1.0, max_time)],
            "showgrid": True,
            "gridcolor": "rgba(140,155,175,0.16)",
            "zeroline": False,
        },
        yaxis={"categoryorder": "array", "categoryarray": tracks, "fixedrange": True},
        bargap=0.36,
        hoverlabel={"bgcolor": "#17212b", "font_color": "#f4f7fa"},
    )
    return figure