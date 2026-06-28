# Tennis Serve Analysis Project

## Project Goal
User uploads a video of their tennis serve. The app analyzes their technique
using pose estimation and generates a personalized training plan based on
the analysis.

## Current Status
**All 4 phases complete ✓**

## Roadmap

### Phase 1: Pose extraction + rule-based analysis (current focus)
- Extract MediaPipe pose landmarks from a serve video
- Identify the ball-contact frame (the key moment to analyze)
- Extract joint angles/metrics at contact: elbow extension, shoulder
  rotation, contact height relative to body, knee bend, toss position
- Compare metrics against known-good ranges (rule-based thresholds —
  no ML/classifier yet)
- Output: structured findings, e.g.
  `{"elbow_extension": 142, "flag": "low - target 160-175"}`
- **No LLM calls in this phase.** Keep this purely deterministic CV/engineering.

### Phase 2: LLM-generated feedback
- Take Phase 1's structured metrics + flags
- Use the Claude API (tool use pattern) to turn flagged issues into clear,
  encouraging coaching language

### Phase 3: RAG-grounded training plan
- Curate a small knowledge base of drill descriptions / coaching tips
- Embed and retrieve drills relevant to the specific flaws detected
- Generate a personalized training plan grounded in the retrieved content
  (not just the model's own memory)

### Phase 4: MCP server wrapper
- Wrap `analyze_pose`, `generate_feedback`, and `generate_training_plan`
  as MCP tools
- Build a small client/agent that chains them in sequence
  (detect → feedback → plan)
- This is the standalone MCP server portfolio piece

## Tech Stack
- Python
- MediaPipe (pose estimation)
- OpenCV (video / frame handling)
- Anthropic Python SDK (Phase 2 onward)
- `uv` for virtual environment / package management

## Constraints & Notes
- Film test videos from a consistent **side-on angle** — joint-angle
  accuracy depends heavily on camera angle
- Don't add LLM/agent decision-making into Phase 1. Pose extraction and
  rule checks should stay deterministic and fast
- Validate visually before trusting any metric: overlay pose keypoints on
  frames and inspect them before building analysis logic on top
- Finding the ball-contact frame is the hardest subproblem in Phase 1 —
  treat it as its own task, not an afterthought


## Coding Preferences
- Prefer targeted, minimal line-level fixes over full rewrites when
  editing existing code
- Surface multiple approaches before implementing when a design decision
  has real tradeoffs (e.g., contact-frame detection method)

## Known Limitations
- `toss_height_ratio` and `contact_height_ratio` thresholds are placeholder/
  unvalidated. They were set loosely enough to not flag the one test serve
  used during development — not derived from real biomechanics data. Treat
  any "good" verdict from these two as decorative for now, not a real signal.
- `elbow_extension` and `knee_bend` thresholds are grounded in tennis
  biomechanics research, but double-check the flexion-vs-extension angle
  convention against the actual calculation before fully trusting absolute
  values.
- `drop_elbow_angle` has no target set — informational only, no pass/fail.