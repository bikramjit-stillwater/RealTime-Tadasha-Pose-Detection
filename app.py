import os
import time
import threading
from collections import deque

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

from src.pose_detector import PoseDetector
from src.pose_utils import get_pose_features, get_step_visibility
from src.pose_scorer import score_tadasana
from src.feedback_engine import get_gemini_feedback, get_rule_based_feedback


SCORE_SMOOTHING_WINDOW = 10
ISSUES_REFRESH_SEC = 4.0
FEEDBACK_REFRESH_SEC = 6.0

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


st.set_page_config(page_title="Tadasana Pose Test", layout="wide")

st.markdown("""
<style>
.big-cam-wrap label[data-testid="stCheckbox"] {
    width: 100% !important;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 12px !important;
    padding: 0.85rem 1.25rem !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    cursor: pointer;
    transition: all 0.15s ease;
}
.big-cam-wrap label[data-testid="stCheckbox"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
}
.big-cam-wrap [data-testid="stCheckbox"] > label {
    width: 100%;
    align-items: center !important;
}
.big-cam-wrap [data-testid="stCheckbox"] p {
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    margin: 0 !important;
}
.big-cam-wrap [data-testid="stCheckbox"] input[type="checkbox"] {
    transform: scale(1.4);
    accent-color: white;
    margin-right: 0.6rem !important;
}
.big-cam-wrap.is-on label[data-testid="stCheckbox"] {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35);
}
.step-score-line {
    font-size: 0.92rem;
    padding: 0.3rem 0;
    color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
}
.step-score-line:last-child { border-bottom: none; }
.step-score-line .num {
    font-weight: 600;
    color: #0f172a;
    float: right;
}
.step-score-line.pass .num { color: #10b981; }
.step-score-line.fail .num { color: #ef4444; }
.step-score-line.notvis .num { color: #9ca3af; }
.step-score-block {
    background: #f8fafc;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


def _render_step_scores(part_scores, steps):
    by_step = {s["step"]: s for s in steps} if steps else {}
    items = sorted(part_scores.items(), key=lambda kv: int(kv[0].split("_")[0]))

    rows = []
    for key, score in items:
        step_num = int(key.split("_")[0])
        raw_name = key.split("_", 1)[1].replace("_", " ")
        pretty_name = (
            raw_name
            .replace("Shoulders Arms", "Shoulders & Arms")
            .replace("Legs Knees", "Legs & Knees")
            .replace("Head Neck", "Head & Neck")
        )

        s = by_step.get(step_num, {})
        not_vis = s.get("not_visible", False)
        passed = s.get("passed", False)

        if not_vis:
            cls = "notvis"
            display = "not visible"
        elif passed:
            cls = "pass"
            display = f"{score}/100"
        else:
            cls = "fail"
            display = f"{score}/100"

        rows.append(
            f"<div class='step-score-line {cls}'>"
            f"Step {step_num} - {pretty_name}"
            f"<span class='num'>{display}</span>"
            f"</div>"
        )

    return "<div class='step-score-block'>" + "".join(rows) + "</div>"


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.score_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)
        self.last_issues_refresh = 0.0
        self.last_feedback_refresh = 0.0
        self.displayed_issues = ["Stand where your full body is visible to the camera."]
        self.displayed_feedback = ["Stand tall and begin the pose."]
        self.displayed_parts_html = ""
        self.latest_score = 0
        self.latest_steps = []
        self.latest_part_scores = {}
        self.pose_detected = False


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector()
        self.state = SharedState()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        display_frame = img.copy()
        now = time.time()

        if self.state is None:
            return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

        results = self.detector.process_frame(img)

        with self.state.lock:
            if results.pose_landmarks:
                self.state.pose_detected = True
                display_frame = self.detector.draw_landmarks(display_frame, results)
                landmarks = results.pose_landmarks.landmark

                h, w = img.shape[:2]
                features = get_pose_features(landmarks, w, h)
                visibility = get_step_visibility(landmarks)
                scoring = score_tadasana(features, visibility)

                self.state.score_buffer.append(scoring["score"])
                smoothed_score = round(sum(self.state.score_buffer) / len(self.state.score_buffer))
                self.state.latest_score = smoothed_score
                self.state.latest_steps = scoring.get("steps", [])
                self.state.latest_part_scores = scoring["part_scores"]

                if now - self.state.last_issues_refresh > ISSUES_REFRESH_SEC:
                    self.state.displayed_issues = scoring["issues"]
                    self.state.displayed_parts_html = _render_step_scores(
                        scoring["part_scores"],
                        scoring.get("steps", []),
                    )
                    self.state.last_issues_refresh = now

                if now - self.state.last_feedback_refresh > FEEDBACK_REFRESH_SEC:
                    self.state.displayed_feedback = get_gemini_feedback(
                        smoothed_score,
                        scoring["issues"],
                    )
                    self.state.last_feedback_refresh = now
            else:
                self.state.pose_detected = False
                self.state.score_buffer.clear()
                self.state.latest_score = 0
                self.state.latest_steps = []
                self.state.latest_part_scores = {}
                self.state.displayed_issues = ["No pose detected"]
                self.state.displayed_feedback = ["Stand where your full body is visible to the camera."]
                self.state.displayed_parts_html = ""

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


st.title("Tadasana Pose Detection - Real Time")
st.write(
    "Live webcam pose detection scored with the same 6-step Tadasana validator "
    "used in the upload page. Stand back so your full body is in the frame."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Reference Video")
    video_path = "assets/tadasana.mp4"
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning("Put your Tadasana video at assets/tadasana.mp4")

with right_col:
    is_on = st.session_state.get("cam_and_feedback", False)
    wrap_class = "big-cam-wrap is-on" if is_on else "big-cam-wrap"

    st.markdown(f"<div class='{wrap_class}'>", unsafe_allow_html=True)
    cam_on = st.checkbox(
        "🎥  Start Cam and Feedback" if not is_on else "⏹  Stop Cam and Feedback",
        key="cam_and_feedback",
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Live Detection")

    score_placeholder = st.empty()
    issues_placeholder = st.empty()
    feedback_placeholder = st.empty()
    parts_placeholder = st.empty()

    if cam_on:
        ctx = webrtc_streamer(
            key="tadasana-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

        if ctx and ctx.state.playing:
            st.success("Webcam is running.")
        else:
            st.info("Allow browser camera access, then click START in the webcam panel.")

        if ctx and ctx.video_processor:
            state = ctx.video_processor.state

            with state.lock:
                if state.pose_detected:
                    score_placeholder.metric("Pose Score", f"{state.latest_score}/100")
                    issues_placeholder.markdown(
                        "### Detected Issues\n" +
                        "\n".join(f"- {x}" for x in state.displayed_issues)
                    )
                    feedback_placeholder.markdown(
                        "### Feedback\n" +
                        "\n".join(f"- {x}" for x in state.displayed_feedback)
                    )
                    parts_placeholder.markdown(
                        "### Step Scores\n" + state.displayed_parts_html,
                        unsafe_allow_html=True,
                    )
                else:
                    score_placeholder.metric("Pose Score", "0/100")
                    issues_placeholder.markdown("### Detected Issues\n- No pose detected")
                    feedback_placeholder.markdown(
                        "### Feedback\n- Stand where your full body is visible to the camera."
                    )
                    parts_placeholder.empty()
        else:
            score_placeholder.metric("Pose Score", "0/100")
            issues_placeholder.markdown(
                "### Detected Issues\n- Waiting for webcam stream."
            )
            feedback_placeholder.markdown(
                "### Feedback\n- Allow browser camera access and start the stream."
            )
            parts_placeholder.empty()
    else:
        st.info("Click **Start Cam and Feedback** above to begin.")
