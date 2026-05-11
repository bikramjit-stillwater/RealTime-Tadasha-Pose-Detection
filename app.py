import os
import time
import threading
from collections import deque

import av
import cv2
import streamlit as st
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


from src.pose_detector import PoseDetector
from src.pose_utils import get_pose_features, get_step_visibility
from src.pose_scorer import score_tadasana
from src.feedback_engine import get_gemini_feedback


SCORE_SMOOTHING_WINDOW = 10
ISSUES_REFRESH_SEC = 4.0
FEEDBACK_REFRESH_SEC = 6.0


st.set_page_config(page_title="Tadasana Pose Test", layout="wide")


st.markdown("""
<style>
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


@st.cache_resource
def get_ice_servers():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid and auth_token:
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers

    return [{"urls": ["stun:stun.l.google.com:19302"]}]


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
                self.state.displayed_issues = ["No pose detected"]
                self.state.displayed_feedback = ["Stand where your full body is visible to the camera."]
                self.state.displayed_parts_html = ""

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


st.title("Tadasana Pose Detection - Real Time")
st.write("Allow camera permission, then click START in the webcam widget below.")

video_path = "assets/tadasana.mp4"
if os.path.exists(video_path):
    st.video(video_path)

ctx = webrtc_streamer(
    key="tadasana-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

score_placeholder = st.empty()
issues_placeholder = st.empty()
feedback_placeholder = st.empty()
parts_placeholder = st.empty()

if ctx and ctx.state.playing:
    st.success("Camera stream is active.")

    if ctx.video_processor:
        state = ctx.video_processor.state
        time.sleep(0.5)

        with state.lock:
            if state.pose_detected:
                score_placeholder.metric("Pose Score", f"{state.latest_score}/100")
                issues_placeholder.markdown(
                    "### Detected Issues\n" + "\n".join(f"- {x}" for x in state.displayed_issues)
                )
                feedback_placeholder.markdown(
                    "### Feedback\n" + "\n".join(f"- {x}" for x in state.displayed_feedback)
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
        "### Detected Issues\n- Click START in the webcam widget and allow browser camera access."
    )
    feedback_placeholder.markdown(
        "### Feedback\n- If the camera still does not start, the connection may require TURN support."
    )
    parts_placeholder.empty()
