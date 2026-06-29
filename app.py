"""
Gradio web interface for the tennis serve analysis pipeline.

Students upload a side-on serve video and receive:
  - Annotated key-frame images (toss zenith, trophy, racket drop, contact)
  - Biomechanical metrics table
  - Coaching feedback grouped by priority
  - Personalised 1-week training plan

Deploy to HuggingFace Spaces (Gradio SDK).
Set ANTHROPIC_API_KEY as a Space secret.
"""

import contextlib
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

import detect_contact as _dc
import analyze_metrics as _am
import generate_feedback as _gf
import generate_training_plan as _gtp


@contextlib.contextmanager
def _suppress_stdout():
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _format_metrics(metrics: dict) -> str:
    metric_labels = {
        "elbow_extension":      "Elbow Extension at Contact",
        "contact_height_ratio": "Contact Height Ratio",
        "knee_bend":            "Knee Bend at Trophy",
        "toss_height_ratio":    "Toss Height Ratio",
        "drop_elbow_angle":     "Drop Elbow Angle (informational)",
    }
    rows = ["| Metric | Value | Status |", "|--------|-------|--------|"]
    for key, m in metrics.items():
        label = metric_labels.get(key, key)
        value = f"{m['value']:.1f} {m['unit']}"
        if m["flag"]:
            status = f"⚠️ {m['flag']}"
        else:
            status = "✅ Good"
        rows.append(f"| {label} | {value} | {status} |")
    return "\n".join(rows)


def _format_feedback(notes: list[dict]) -> str:
    grouped: dict[str, list] = {"high": [], "medium": [], "low": []}
    for note in notes:
        grouped.setdefault(note["priority"], []).append(note)

    icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    titles = {"high": "Priority Fixes", "medium": "Room to Improve", "low": "Doing Well"}
    sections = []
    for level in ("high", "medium", "low"):
        items = grouped.get(level, [])
        if not items:
            continue
        lines = [f"## {icons[level]} {titles[level]}"]
        for n in items:
            lines.append(f"**{n['metric'].replace('_', ' ').title()}**")
            lines.append(f"- *Observation:* {n['observation']}")
            lines.append(f"- *Recommendation:* {n['recommendation']}")
            lines.append("")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def analyze(video_path: str, progress=gr.Progress()) -> tuple:
    _load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise gr.Error("ANTHROPIC_API_KEY is not set. Add it as a Space secret.")

    if video_path is None:
        raise gr.Error("Please upload a serve video first.")

    out_dir = Path(tempfile.mkdtemp())
    import shutil
    suffix = Path(video_path).suffix
    tmp_video = out_dir / f"serve{suffix}"
    shutil.copy(video_path, tmp_video)
    video_str = str(tmp_video)

    # Step 1: detect key frames
    progress(0.10, desc="Detecting ball and key frames…")
    with _suppress_stdout():
        candidate_frame, toss_zenith_frame, track = _dc.find_contact_frame(video_str)
        candidate_det = next(((cx, cy) for fi, cx, cy in track if fi == candidate_frame), None)
        ball_cy = candidate_det[1] if candidate_det else 0
        key_frames = _dc.detect_key_frames(video_str, candidate_frame, ball_cy)
        key_frames["toss_zenith"] = toss_zenith_frame

    # Step 2: save annotated key-frame images
    progress(0.30, desc="Saving key-frame images…")
    with _suppress_stdout():
        _dc.save_outputs(video_str, key_frames, track, ball_pos=candidate_det)

    frame_images = []
    for key in ("toss_zenith", "trophy", "drop", "contact"):
        stem = tmp_video.stem
        img_path = out_dir / f"{stem}_{key}.jpg"
        if img_path.exists():
            frame_images.append((str(img_path), key.replace("_", " ").title()))

    # Step 3: extract metrics
    progress(0.45, desc="Extracting biomechanical metrics…")
    with _suppress_stdout():
        metrics = _am.analyze_metrics(video_str, key_frames)
    metrics_md = _format_metrics(metrics)

    # Step 4: generate coaching feedback
    progress(0.65, desc="Generating coaching feedback…")
    with _suppress_stdout():
        notes = _gf.generate_feedback(metrics)
    feedback_md = _format_feedback(notes)

    # Step 5: retrieve drills + generate training plan
    progress(0.80, desc="Building your 1-week training plan…")
    grouped = {"high": [], "medium": [], "low": []}
    for note in notes:
        grouped.setdefault(note["priority"], []).append(note)
    with _suppress_stdout():
        retrieved = _gtp.retrieve_drills(grouped)
        plan_text = _gtp.generate_plan(grouped, retrieved)

    progress(1.0, desc="Done!")
    return frame_images, metrics_md, feedback_md, plan_text


with gr.Blocks(title="Tennis Serve Analyzer") as demo:
    gr.Markdown("""
# 🎾 Tennis Serve Analyzer
Upload a **side-on video** of your tennis serve. The app will analyze your technique
using pose estimation and generate a personalised coaching report and 1-week training plan.

**Tips for best results:**
- Film from the side (perpendicular to the baseline)
- Include at least one full serve from start to follow-through
- Good lighting and a steady camera help accuracy
""")

    with gr.Row():
        video_input = gr.Video(label="Upload Serve Video", height=400)

    analyze_btn = gr.Button("Analyze My Serve ▶", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("📸 Key Frames"):
            gallery = gr.Gallery(
                label="Key Moments Detected",
                columns=2,
                height=500,
                object_fit="contain",
            )
        with gr.Tab("📊 Metrics"):
            metrics_out = gr.Markdown(label="Biomechanical Metrics")
        with gr.Tab("💬 Coaching Feedback"):
            feedback_out = gr.Markdown(label="Coaching Notes")
        with gr.Tab("📅 Training Plan"):
            plan_out = gr.Markdown(label="1-Week Training Plan")

    analyze_btn.click(
        fn=analyze,
        inputs=video_input,
        outputs=[gallery, metrics_out, feedback_out, plan_out],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
