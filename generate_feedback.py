"""
Phase 2: LLM-generated coaching feedback.

Reads a _metrics.json file produced by analyze_metrics.py, calls Claude
using the tool-use pattern, and collects one coaching note per metric.

Usage:
    python generate_feedback.py IMG_3961_metrics.json

Output:
    Prints notes grouped by priority.
    Saves <stem>_feedback.json in the same directory.

Requires ANTHROPIC_API_KEY in the environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

_MODEL = "claude-sonnet-4-6"


def _load_dotenv() -> None:
    """Read KEY=value lines from .env next to this script into os.environ."""
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

_SYSTEM_PROMPT = """\
You are an encouraging tennis coach reviewing biomechanical measurements from a serve video.
For each metric provided, call add_coaching_note exactly once.

Metric reference — what each measures and what good looks like:
- elbow_extension: racket-arm elbow angle at ball contact. Target ≥160°. Higher means \
more arm extension and more power. 178° is excellent.
- knee_bend: racket-side knee angle at trophy position. Target 130–155°. A lower angle \
means more knee flex and more leg drive. 169° means nearly straight legs — the main \
area to work on.
- drop_elbow_angle: elbow angle at the racket drop (backscratch). No validated target \
range — describe what it suggests about the backswing depth descriptively.
- contact_height_ratio: wrist height relative to body height at contact. The threshold \
for this metric is NOT validated — mention the value positively or neutrally, do not \
flag it as an issue.
- toss_height_ratio: toss wrist height above shoulder relative to body height. The \
threshold for this metric is NOT validated — mention the value positively or neutrally, \
do not flag it as an issue.

Priority rules:
- high: metric is flagged AND its threshold is validated (only elbow_extension and \
knee_bend qualify)
- medium: no flag but clear room to improve on a validated metric
- low: performing well, or metric is informational/unvalidated

Tone: specific, encouraging, and actionable. Keep each field to one sentence.\
"""

_ADD_COACHING_NOTE_TOOL = {
    "name": "add_coaching_note",
    "description": "Record a coaching observation for one serve metric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "The metric name exactly as in the input JSON.",
            },
            "observation": {
                "type": "string",
                "description": "What this number reveals about the player's technique.",
            },
            "recommendation": {
                "type": "string",
                "description": "A specific, actionable coaching cue the player can practise.",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "high = flagged issue on a validated metric; "
                    "medium = room to improve; "
                    "low = doing well or metric is informational/unvalidated."
                ),
            },
        },
        "required": ["metric", "observation", "recommendation", "priority"],
    },
}

# Human-readable descriptions injected alongside each metric value.
_METRIC_DESCRIPTIONS = {
    "elbow_extension":      "racket-arm elbow angle at contact",
    "knee_bend":            "racket-side knee angle at trophy position",
    "drop_elbow_angle":     "elbow angle at racket drop (backscratch)",
    "contact_height_ratio": "wrist height / body height at contact",
    "toss_height_ratio":    "toss wrist height above shoulder / body height",
}


def _build_user_message(metrics: dict) -> str:
    lines = ["Serve analysis metrics:\n"]
    for name, m in metrics.items():
        desc = _METRIC_DESCRIPTIONS.get(name, name)
        value_str = f"{m['value']} {m['unit']}"
        if m["flag"]:
            status = f"FLAG — \"{m['flag']}\""
        elif name in ("contact_height_ratio", "toss_height_ratio"):
            status = "GOOD (threshold unvalidated)"
        elif name == "drop_elbow_angle":
            status = "INFORMATIONAL (no target range)"
        else:
            status = "GOOD (no flag)"
        lines.append(f"{name}: {value_str} — {desc}")
        lines.append(f"  status: {status}\n")
    return "\n".join(lines)


def generate_feedback(metrics: dict) -> list[dict]:
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": _build_user_message(metrics)}]
    notes: list[dict] = []

    while True:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_ADD_COACHING_NOTE_TOOL],
            messages=messages,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "add_coaching_note":
                notes.append(block.input)

        if response.stop_reason == "end_turn":
            break

        # stop_reason == "tool_use" — send tool results back and continue
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": "noted"}
            for b in response.content
            if b.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})

    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM coaching feedback from serve metrics.")
    parser.add_argument("metrics_json", help="Path to _metrics.json from analyze_metrics.py")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.exists():
        sys.exit(f"File not found: {metrics_path}")

    with open(metrics_path) as f:
        payload = json.load(f)

    metrics = payload["metrics"]
    print(f"Loaded {len(metrics)} metrics from {metrics_path.name}")
    print("Calling Claude …\n")

    notes = generate_feedback(metrics)

    # Group by priority
    grouped: dict[str, list] = {"high": [], "medium": [], "low": []}
    for note in notes:
        grouped.setdefault(note["priority"], []).append(note)

    print("=== Coaching Feedback ===\n")
    for level in ("high", "medium", "low"):
        items = grouped.get(level, [])
        if not items:
            continue
        label = {"high": "Priority fixes", "medium": "Room to improve", "low": "Doing well"}[level]
        print(f"[{label.upper()}]")
        for n in items:
            print(f"  {n['metric']}")
            print(f"    Observation:     {n['observation']}")
            print(f"    Recommendation:  {n['recommendation']}")
        print()

    out_path = metrics_path.parent / metrics_path.name.replace("_metrics.json", "_feedback.json")
    with open(out_path, "w") as f:
        json.dump(grouped, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
