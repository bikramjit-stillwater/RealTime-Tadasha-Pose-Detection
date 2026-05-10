"""
Real-time Tadasana pose detection app.

Uses the same 6-step scoring logic as the static website:
  1. Stance       2. Body Balance     3. Legs & Knees
  4. Spine        5. Shoulders & Arms 6. Head & Neck

Visibility-aware: if a body part is not visible, that step scores 0
with a clear message. Compound penalties are applied when multiple
steps fail.

Performance / UX:
  - Score is smoothed across the last N frames so it doesn't flicker.
  - Issues + suggestions refresh on a slower cadence so the user has
    time to read and react.
"""

import os
import time
from collections import deque

import cv2
import streamlit as st

from src.pose_detector import PoseDetector
from src.pose_utils import get_pose_features, get_step_visibility
from src.pose_scorer import score_tadasana
from src.feedback_engine import get_gemini_feedback, get_rule_based_feedback


# -----------------------------------------------------------------------------
# Tuning knobs - adjust the "speed" of feedback here
# -----------------------------------------------------------------------------
SCORE_SMOOTHING_WINDOW = 10   # number of frames to average score over
ISSUES_REFRESH_SEC = 4.0      # how often the on-screen "issues" list updates
FEEDBACK_REFRESH_SEC = 6.0    # how often the suggestion text updates (slower)
# -----------------------------------------------------------------------------


st.set_page_config(page_title="Tadasana Pose Test", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Big toggle-button styling for the Start Cam control.
   We target the wrapper that has class "big-cam-btn" applied via st.container. */
div[data-testid="stCheckbox"].big-cam-btn-marker { display: none; }

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

/* When toggled ON, switch the gradient to indicate "running" */
.big-cam-wrap.is-on label[data-testid="stCheckbox"] {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35);
}

/* Step-score readable list */
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

# ---- Right column: big single button at the top, then heading ----
with right_col:
    # Need to know toggle state BEFORE rendering so we can apply the right CSS class
    is_on = st.session_state.get("cam_and_feedback", False)
    wrap_class = "big-cam-wrap is-on" if is_on else "big-cam-wrap"

    st.markdown(f"<div class='{wrap_class}'>", unsafe_allow_html=True)
    cam_on = st.checkbox(
        "🎥  Start Cam and Feedback" if not is_on else "⏹  Stop Cam and Feedback",
        key="cam_and_feedback",
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # One toggle controls both: webcam runs AND Gemini feedback is on
    run = cam_on
    use_gemini = cam_on

    st.subheader("Live Detection")

    frame_placeholder = st.empty()
    score_placeholder = st.empty()
    issues_placeholder = st.empty()
    feedback_placeholder = st.empty()
    parts_placeholder = st.empty()

detector = PoseDetector()


def _render_step_scores(part_scores, steps):
    """Build readable HTML for the step-by-step scores (no JSON)."""
    by_step = {s["step"]: s for s in steps} if steps else {}

    items = sorted(part_scores.items(), key=lambda kv: int(kv[0].split("_")[0]))

    rows = []
    for key, score in items:
        step_num = int(key.split("_")[0])
        raw_name = key.split("_", 1)[1].replace("_", " ")
        # Restore "&" for the compound names
        pretty_name = (raw_name
                       .replace("Shoulders Arms", "Shoulders & Arms")
                       .replace("Legs Knees", "Legs & Knees")
                       .replace("Head Neck", "Head & Neck"))

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


if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam could not be opened.")
    else:
        score_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)

        last_issues_refresh = 0.0
        last_feedback_refresh = 0.0

        displayed_issues = ["Stand where your full body is visible to the camera."]
        displayed_feedback = ["Stand tall and begin the pose."]
        displayed_parts_html = ""

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            results = detector.process_frame(frame)
            display_frame = frame.copy()

            now = time.time()

            if results.pose_landmarks:
                display_frame = detector.draw_landmarks(display_frame, results)
                landmarks = results.pose_landmarks.landmark

                h, w = frame.shape[:2]
                features = get_pose_features(landmarks, w, h)
                visibility = get_step_visibility(landmarks)
                scoring = score_tadasana(features, visibility)

                # Smooth the score across recent frames (prevents flicker)
                score_buffer.append(scoring["score"])
                smoothed_score = round(sum(score_buffer) / len(score_buffer))

                score_placeholder.metric("Pose Score", f"{smoothed_score}/100")

                # Refresh issues list on a slower cadence so the user can read
                if now - last_issues_refresh > ISSUES_REFRESH_SEC:
                    displayed_issues = scoring["issues"]
                    displayed_parts_html = _render_step_scores(
                        scoring["part_scores"], scoring.get("steps", [])
                    )
                    last_issues_refresh = now

                issues_placeholder.markdown(
                    "### Detected Issues\n" +
                    "\n".join(f"- {x}" for x in displayed_issues)
                )

                # Refresh suggestion text even more slowly (especially for Gemini)
                if now - last_feedback_refresh > FEEDBACK_REFRESH_SEC:
                    if use_gemini:
                        displayed_feedback = get_gemini_feedback(
                            smoothed_score, scoring["issues"]
                        )
                    else:
                        displayed_feedback = get_rule_based_feedback(scoring["issues"])
                    last_feedback_refresh = now

                feedback_placeholder.markdown(
                    "### Feedback\n" +
                    "\n".join(f"- {x}" for x in displayed_feedback)
                )

                parts_placeholder.markdown(
                    "### Step Scores\n" + displayed_parts_html,
                    unsafe_allow_html=True,
                )

            else:
                score_buffer.clear()
                score_placeholder.metric("Pose Score", "0/100")
                issues_placeholder.markdown(
                    "### Detected Issues\n- No pose detected"
                )
                feedback_placeholder.markdown(
                    "### Feedback\n- Stand where your full body is visible to the camera."
                )
                parts_placeholder.empty()

            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                display_frame, channels="RGB", use_container_width=True
            )

            run = st.session_state.get("cam_and_feedback", run)

        cap.release()
else:
    st.info("Click **Start Cam and Feedback** above to begin.")