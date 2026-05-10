# Tadasana Pose Detection Test App

## Setup

1. Create a virtual environment
2. Install dependencies
3. Add your yoga reference video at `assets/tadasana.mp4`
4. Add Gemini API key in `.env` if needed

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Current features

- Reference Tadasana video
- Live webcam pose detection
- MediaPipe skeleton overlay
- Basic Tadasana score
- Rule-based feedback
- Optional Gemini 2.5 Flash feedback

## First goal

Use this app only to verify:
- webcam opens,
- pose is detected,
- skeleton appears,
- score changes when posture changes.