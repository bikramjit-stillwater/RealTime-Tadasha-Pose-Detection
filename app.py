"""
Real-time Tadasana pose detection (Streamlit Cloud + Twilio TURN).

Key design choices that make this actually work on the cloud:
  - recv() is LIGHTWEIGHT: only pose detection + scoring + landmark drawing.
    It does NOT call Gemini (would block the frame loop).
  - Main Streamlit thread polls the processor's shared state in a loop
    and updates the UI placeholders. Gemini is called here, on a slow
    cadence, so the video stream is never blocked.
"""

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
from src.feedback_engine import get_gemini_feedback, get_rule_based_feedback


# -----------------------------------------------------------------------------
# Tuning knobs
# -----------------------------------------------------------------------------
SCORE_SMOOTHING_WINDOW = 10
ISSUES_REFRESH_SEC = 4.0
FEEDBACK_REFRESH_SEC = 6.0
UI_POLL_INTERVAL_SEC = 0.5
# -----------------------------------------------------------------------------


st.set_page_config(page_title="Tadasana Pose Test", layout="wide")

st.markdown("""
<style>
.step-score-line {
    font-size: 0.92rem; padding: 0.3rem 0; color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
}
.step-score-line:last-child { border-bottom: none; }
.step-score-line .num { font-weight: 600; color: #0f172a; float: right; }
.step-score-line.pass .num { color: #10b981; }
.step-score-line.fail .num { color: #ef4444; }
.step-score-line.notvis .num { color: #9ca3af; }
.step-score-block {
    background: #f8fafc; border-radius: 8px; padding: 0.5rem 0.9rem;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Twilio TURN servers (cached - one API call per session)
# =============================================================================
@st.cache_resource
def get_ice_servers():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid and auth_token:
        try:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            st.warning(f"Twilio TURN failed, falling back to STUN only: {e}")

    return [{"urls": ["stun:stun.l.google.com:19302"]}]


# =============================================================================
# Video processor - LIGHTWEIGHT: only pose detection + scoring + drawing.
# Stores the latest scoring result in self.latest (thread-safe).
# Does NOT call Gemini or update Streamlit UI directly.
# =============================================================================
class TadasanaProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector()
        self.score_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)
        self.lock = threading.Lock()
        self.latest = {
            "score": 0,
            "issues": [],
            "part_scores": {},
            "steps": [],
            "has_pose": False,
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        display = img.copy()

        try:
            results = self.detector.process_frame(img)

            if results.pose_landmarks:
                display = self.detector.draw_landmarks(display, results)
                landmarks = results.pose_landmarks.landmark
                h, w = img.shape[:2]

                features = get_pose_features(landmarks, w, h)
                visibility = get_step_visibility(landmarks)
                scoring = score_tadasana(features, visibility)

                self.score_buffer.append(scoring["score"])
                smoothed = round(sum(self.score_buffer) / len(self.score_buffer))

                with self.lock:
                    self.latest = {
                        "score": smoothed,
                        "issues": scoring["issues"],
                        "part_scores": scoring["part_scores"],
                        "steps": scoring.get("steps", []),
                        "has_pose": True,
                    }
            else:
                with self.lock:
                    self.latest["has_pose"] = False

            return av.VideoFrame.from_ndarray(display, format="bgr24")

        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")


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
            cls, display = "notvis", "not visible"
        elif passed:
            cls, display = "pass", f"{score}/100"
        else:
            cls, display = "fail", f"{score}/100"

        rows.append(
            f"<div class='step-score-line {cls}'>"
            f"Step {step_num} - {pretty_name}"
            f"<span class='num'>{display}</span>"
            f"</div>"
        )
    return "<div class='step-score-block'>" + "".join(rows) + "</div>"


# =============================================================================
# UI
# =============================================================================
st.title("Tadasana Pose Detection - Real Time")
st.write("Allow camera permission, then click START in the webcam widget below.")

use_gemini = st.checkbox("Use Gemini feedback (slower, smarter cues)", value=False)

video_path = "assets/tadasana.mp4"
if os.path.exists(video_path):
    with st.expander("Reference video"):
        st.video(video_path)

ctx = webrtc_streamer(
    key="tadasana-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False,
    },
    video_processor_factory=TadasanaProcessor,
    async_processing=True,
)

score_placeholder = st.empty()
issues_placeholder = st.empty()
feedback_placeholder = st.empty()
parts_placeholder = st.empty()


# =============================================================================
# Main-thread polling loop - reads processor.latest, updates UI, calls Gemini.
# This is what was MISSING in your previous version - without this loop, the
# UI placeholders stay frozen at their initial values.
# =============================================================================
if ctx.state.playing and ctx.video_processor:
    last_issues_refresh = 0.0
    last_feedback_refresh = 0.0

    displayed_issues = ["Getting into position..."]
    displayed_feedback = ["Stand tall and begin the pose."]
    displayed_parts_html = ""

    while ctx.state.playing:
        # Snapshot the latest scoring result from the processor thread
        with ctx.video_processor.lock:
            data = dict(ctx.video_processor.latest)

        now = time.time()

        if data.get("has_pose"):
            score_placeholder.metric("Pose Score", f"{data['score']}/100")

            # Issues + step scores refresh on a slow cadence
            if now - last_issues_refresh > ISSUES_REFRESH_SEC:
                displayed_issues = data["issues"] or ["Good Tadasana alignment detected"]
                displayed_parts_html = _render_step_scores(
                    data["part_scores"], data["steps"]
                )
                last_issues_refresh = now

            issues_placeholder.markdown(
                "### Detected Issues\n" +
                "\n".join(f"- {x}" for x in displayed_issues)
            )

            parts_placeholder.markdown(
                "### Step Scores\n" + displayed_parts_html,
                unsafe_allow_html=True,
            )

            # Feedback refresh even slower (especially for Gemini)
            if now - last_feedback_refresh > FEEDBACK_REFRESH_SEC:
                if use_gemini:
                    displayed_feedback = get_gemini_feedback(
                        data["score"], data["issues"]
                    )
                else:
                    displayed_feedback = get_rule_based_feedback(data["issues"])
                last_feedback_refresh = now

            feedback_placeholder.markdown(
                "### Feedback\n" +
                "\n".join(f"- {x}" for x in displayed_feedback)
            )

        else:
            score_placeholder.metric("Pose Score", "0/100")
            issues_placeholder.markdown(
                "### Detected Issues\n- No pose detected - step back so your full body fits in frame"
            )
            feedback_placeholder.markdown(
                "### Feedback\n- Stand where your full body is visible to the camera."
            )
            parts_placeholder.empty()

        time.sleep(UI_POLL_INTERVAL_SEC)
else:
    score_placeholder.metric("Pose Score", "0/100")
    issues_placeholder.markdown(
        "### Detected Issues\n- Click START in the webcam widget and allow browser camera access."
    )
    feedback_placeholder.markdown(
        "### Feedback\n- Once the stream is live, scores and feedback will appear here."
    )
    parts_placeholder.empty()
