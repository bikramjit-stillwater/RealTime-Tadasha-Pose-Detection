"""
Feature extraction for real-time Tadasana scoring.

Uses the SAME feature set as the static website's 6-step validator:
  - stance_ratio       (Step 1)
  - body_lean          (Step 2)
  - knee bend angles   (Step 3)
  - spine_tilt         (Step 4)
  - arm drop / elbow / closeness / symmetry  (Step 5)
  - head_offset        (Step 6)

Also exposes a visibility check per step so missing body parts score 0.
"""

import math
import numpy as np


# -----------------------------------------------------------------------------
# MediaPipe landmark indices
# -----------------------------------------------------------------------------
POSE_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# Critical landmarks per step - if any are not visible, that step scores 0
STEP_CRITICAL_LANDMARKS = {
    1: ["left_ankle", "right_ankle", "left_hip", "right_hip"],                       # Stance
    2: ["left_shoulder", "right_shoulder", "left_hip", "right_hip",                  # Body Balance
        "left_ankle", "right_ankle"],
    3: ["left_hip", "right_hip", "left_knee", "right_knee",                          # Legs & Knees
        "left_ankle", "right_ankle"],
    4: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],                 # Spine
    5: ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",              # Shoulders & Arms
        "left_wrist", "right_wrist"],
    6: ["nose", "left_shoulder", "right_shoulder"],                                  # Head & Neck
}

VISIBILITY_THRESHOLD = 0.5


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def landmark_to_px(landmarks, idx, w, h):
    """Convert normalized landmark to pixel coords."""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def calculate_angle(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def angle_from_vertical(p_top, p_bottom):
    """Angle (deg) of the line p_bottom -> p_top measured from the vertical axis."""
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# -----------------------------------------------------------------------------
# Visibility check
# -----------------------------------------------------------------------------
def landmark_is_visible(lms, idx):
    """Visible if MediaPipe confidence >= threshold AND coords inside the image."""
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESHOLD:
        return False
    if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
        return False
    return True


def get_step_visibility(lms):
    """Return {step_num: bool} - True if all critical landmarks for that step are visible."""
    result = {}
    for step_num, names in STEP_CRITICAL_LANDMARKS.items():
        all_visible = True
        for name in names:
            idx = POSE_LANDMARKS[name]
            if not landmark_is_visible(lms, idx):
                all_visible = False
                break
        result[step_num] = all_visible
    return result


# -----------------------------------------------------------------------------
# Feature extraction (matches the static site's build_features)
# -----------------------------------------------------------------------------
def get_pose_features(landmarks, w, h):
    """Build the rich feature dict expected by the 6-step validator.

    Args:
        landmarks: MediaPipe pose landmarks list
        w, h: frame width / height in pixels
    """
    p = lambda name: landmark_to_px(landmarks, POSE_LANDMARKS[name], w, h)

    ls = p("left_shoulder");  rs = p("right_shoulder")
    lh = p("left_hip");       rh = p("right_hip")
    lel = p("left_elbow");    rel = p("right_elbow")
    lw_pt = p("left_wrist");  rw_pt = p("right_wrist")
    lk = p("left_knee");      rk = p("right_knee")
    la = p("left_ankle");     ra = p("right_ankle")
    nose = p("nose")

    # Step 2 - body lean: how far the body's vertical center is offset from ankle center
    body_center_x = ((ls[0] + rs[0]) / 2 + (lh[0] + rh[0]) / 2) / 2
    ankle_center_x = (la[0] + ra[0]) / 2
    body_lean = abs(body_center_x - ankle_center_x) / w

    # Step 3 - knee angles
    left_knee_bend = calculate_angle(lh, lk, la)
    right_knee_bend = calculate_angle(rh, rk, ra)

    # Step 1 - stance ratio (ankle distance vs hip distance)
    ankle_distance = abs(la[0] - ra[0])
    hip_distance = abs(lh[0] - rh[0])
    stance_ratio = ankle_distance / hip_distance if hip_distance > 1 else 1.0

    # Step 4 - spine tilt from vertical
    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    spine_tilt = angle_from_vertical(mid_shoulders, mid_hips)

    # Step 6 - head offset from shoulder midpoint
    head_offset = abs(nose[0] - mid_shoulders[0]) / w

    # Step 5 - arm metrics (arms-overhead Tadasana)
    shoulder_y = (ls[1] + rs[1]) / 2
    ankle_y = (la[1] + ra[1]) / 2
    body_height = ankle_y - shoulder_y
    if body_height > 1:
        # Negative drop means wrist is ABOVE shoulder (arms raised overhead)
        left_arm_drop = (lw_pt[1] - shoulder_y) / body_height
        right_arm_drop = (rw_pt[1] - shoulder_y) / body_height
    else:
        left_arm_drop = 0.5
        right_arm_drop = 0.5

    left_elbow_angle = calculate_angle(ls, lel, lw_pt)
    right_elbow_angle = calculate_angle(rs, rel, rw_pt)
    arm_closeness = abs(lw_pt[0] - rw_pt[0]) / w

    # Tilts (kept for diagnostics, not strictly required by validator)
    shoulder_tilt = abs(ls[1] - rs[1]) / h
    hip_tilt = abs(lh[1] - rh[1]) / h

    return {
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "body_lean": body_lean,
        "left_knee_bend": left_knee_bend,
        "right_knee_bend": right_knee_bend,
        "stance_ratio": stance_ratio,
        "spine_tilt": spine_tilt,
        "head_offset": head_offset,
        "left_arm_drop": left_arm_drop,
        "right_arm_drop": right_arm_drop,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "arm_closeness": arm_closeness,
    }