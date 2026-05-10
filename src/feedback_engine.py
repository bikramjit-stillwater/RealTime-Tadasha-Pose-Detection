import os
import google.generativeai as genai


# Mappings for the new 6-step issue strings produced by pose_scorer.py
# Each entry maps an "issue" string (or a substring match) to a friendly cue.
RULE_MAP = {
    # Step 1
    "Feet are too far apart":
        "Bring your feet closer - aim for hip-width or together.",
    # Step 2
    "Body is leaning":
        "Stand evenly - press both feet down equally and stack your weight.",
    # Step 3
    "Knees are locked":
        "Soften your knees - keep them straight but not rigidly locked.",
    "Knees are bent":
        "Gently straighten your knees without locking them.",
    "One knee bent and the other locked":
        "Even out your legs - both knees soft and active.",
    # Step 4
    "Spine is not vertical":
        "Lengthen your spine - tailbone down, crown of the head up.",
    # Step 5 (arms overhead)
    "Arms are not raised":
        "Stretch your arms straight up overhead, palms toward each other.",
    "Reach arms higher":
        "Reach a little higher - fully extend the arms overhead.",
    "Elbows are bent":
        "Straighten your elbows so the arms are fully extended.",
    "Bring the arms closer together":
        "Bring your arms closer together overhead, palms near each other.",
    "One arm is higher":
        "Keep both arms even - lift them to the same height.",
    # Step 6
    "Head is tilting":
        "Keep your head centered between your arms, gaze soft and forward.",
    # Visibility
    "not visible in the frame":
        "Step back so your full body is visible in the camera.",
    # Good
    "Good Tadasana alignment detected":
        "Good alignment - stay grounded and breathe steadily.",
}


def _friendly_cue(issue):
    """Map a raw issue string to a friendly cue. Falls back to the issue itself."""
    for needle, cue in RULE_MAP.items():
        if needle.lower() in issue.lower():
            return cue
    return issue


def get_rule_based_feedback(issues):
    """Produce up to 3 friendly cues based on the detected issues."""
    if not issues:
        return [
            "Stand tall and keep the body relaxed.",
            "Lengthen the spine and reach up through the crown.",
            "Breathe steadily and stay grounded.",
        ]
    return [_friendly_cue(i) for i in issues[:3]]


def get_gemini_feedback(score, issues):
    """Use Gemini if a key is available, else fall back to rule-based."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return get_rule_based_feedback(issues)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "- None significant"

        prompt = f"""
You are a calm, encouraging yoga teacher.
A student is performing Tadasana (Mountain Pose) with arms raised overhead.
Be HONEST - if the score is low, do not call it good.

Score: {score}/100

Score bands (use this tone):
  90-100  Excellent - small refinements only
  75-89   Good - one or two clear corrections
  55-74   Mixed - several real issues, gentle but firm
  30-54   Poor - basics are off, walk them through it
   0-29   Very poor - it is not Tadasana yet

Detected issues:
{issues_text}

Give EXACTLY 3 short corrections, one per line.
Each line must be under 18 words and start with a verb.
Match the tone to the score band.
Do not number the lines, do not use bullets.
"""

        response = model.generate_content(prompt)
        text = (response.text or "").strip()

        lines = [
            line.strip("-•0123456789. ").strip()
            for line in text.split("\n")
            if line.strip()
        ]
        lines = [l for l in lines if l]
        return lines[:3] if lines else get_rule_based_feedback(issues)

    except Exception:
        return get_rule_based_feedback(issues)