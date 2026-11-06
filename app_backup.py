import streamlit as st
import os
import tempfile

from video_engine.scene_detector import detect_scenes, extract_frames


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="DeepTrace",
    page_icon="🔎",
    layout="wide"
)


# -----------------------------
# HEADER
# -----------------------------

st.title("🔎 DeepTrace")
st.subheader("Multimodal AI Deepfake Forensics")

st.write(
    "Upload a video and DeepTrace will analyze its scenes "
    "and extract representative frames for forensic analysis."
)


# -----------------------------
# ENGINE STATUS
# -----------------------------

st.markdown("## Engine Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Video Pipeline")

with col2:
    st.success("✅ Scene Localization")

with col3:
    st.success("✅ Metadata Analysis")

col4, col5, col6 = st.columns(3)

with col4:
    st.warning("⚠️ Audio Model — Plugin Required")

with col5:
    st.warning("⚠️ Deepfake Model — Plugin Required")

with col6:
    st.warning("⚠️ A/V Synchronization — Plugin Required")


st.divider()


# -----------------------------
# VIDEO UPLOAD
# -----------------------------

st.header("🎥 Upload Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv"]
)


# -----------------------------
# PROCESS VIDEO
# -----------------------------

if uploaded_file is not None:

    st.success(f"Video loaded: {uploaded_file.name}")

    # Create temporary video file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as temp_video:

        temp_video.write(uploaded_file.read())
        video_path = temp_video.name


    # -------------------------
    # VIDEO PREVIEW
    # -------------------------

    st.video(uploaded_file)

    st.divider()

    # -------------------------
    # ANALYZE BUTTON
    # -------------------------

    if st.button(
        "🔬 Start Forensic Analysis",
        type="primary"
    ):

        # Create temporary frame directory
        frame_folder = tempfile.mkdtemp(
            prefix="deeptrace_frames_"
        )

        # -------------------------
        # SCENE DETECTION
        # -------------------------

        with st.spinner("Detecting scenes..."):

            try:

                scenes = detect_scenes(video_path)

            except Exception as e:

                st.error(
                    f"Scene detection failed: {e}"
                )

                st.stop()


        st.success(
            f"Detected {len(scenes)} scene(s)."
        )


        # -------------------------
        # DISPLAY SCENES
        # -------------------------

        st.header("🎬 Scene Timeline")

        for scene in scenes:

            start = scene["start"]
            end = scene["end"]

            st.write(
                f"**Scene {scene['scene']}**  "
                f"{start:.2f}s → {end:.2f}s"
            )

            st.progress(
                min(
                    int((end / max(scenes[-1]["end"], 1)) * 100),
                    100
                )
            )


        # -------------------------
        # FRAME EXTRACTION
        # -------------------------

        with st.spinner(
            "Extracting forensic frames..."
        ):

            try:

                frames = extract_frames(
                    video_path,
                    frame_folder,
                    interval=1
                )

            except Exception as e:

                st.error(
                    f"Frame extraction failed: {e}"
                )

                st.stop()


        st.success(
            f"Extracted {len(frames)} frames."
        )


        # -------------------------
        # FRAME GALLERY
        # -------------------------

        st.header("🖼️ Extracted Frames")

        if frames:

            # Show first 12 frames
            display_frames = frames[:12]

            columns = st.columns(4)

            for index, frame in enumerate(
                display_frames
            ):

                with columns[index % 4]:

                    st.image(
                        frame["path"],
                        caption=(
                            f"{frame['timestamp']:.2f}s"
                        ),
                        use_container_width=True
                    )

        else:

            st.warning(
                "No frames were extracted."
            )


        # -------------------------
        # CURRENT ANALYSIS STATUS
        # -------------------------

        st.divider()

        st.header("🧠 Forensic Analysis Status")

        st.info(
            "Video processing and scene analysis "
            "are complete. Deepfake, audio and "
            "audio/video synchronization models "
            "will be connected in the next stages."
        )