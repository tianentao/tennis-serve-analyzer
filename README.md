---
title: Tennis Serve Analyzer
emoji: 🎾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Tennis Serve Analyzer

Upload a side-on video of your tennis serve. The app uses MediaPipe pose estimation
to extract biomechanical metrics, then generates coaching feedback and a personalised
1-week training plan using Claude AI.

## How to use

1. Film your serve from the **side** (perpendicular to the baseline)
2. Upload the video using the button above
3. Click **Analyze My Serve**
4. Review your key frames, metrics, coaching notes, and training plan across the four tabs

## What it measures

| Metric | Target range |
|--------|-------------|
| Elbow extension at contact | ≥ 160° (full extension) |
| Knee bend at trophy position | ≤ 155° (loaded, not straight) |
| Contact height ratio | ≥ 1.1× body height |
| Toss height ratio | 0.2–0.5× body height |
| Drop elbow angle | Informational only |

## Tech stack

- **MediaPipe** — pose landmark extraction
- **OpenCV** — video decoding and key-frame annotation
- **Claude API** — coaching feedback (tool-use) and training plan generation
- **sentence-transformers** — RAG-based drill retrieval from a curated knowledge base
- **Gradio** — web interface
