"""
Phase 3: RAG-grounded training plan generation.

Reads a _feedback.json file from Phase 2, retrieves relevant drills from the
knowledge base using sentence-transformers vector similarity, then calls Claude
to write a personalised 1-week training plan grounded in those drills.

Usage:
    python generate_training_plan.py IMG_3961_feedback.json

Output:
    <stem>_training_plan.txt  — human-readable plan
    <stem>_training_plan.json — structured: {retrieved_drills, plan}

Requires ANTHROPIC_API_KEY in the environment or a .env file in the project root.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = "claude-sonnet-4-6"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_DRILLS_PATH = Path(__file__).parent / "knowledge_base" / "drills.json"
_TOP_K_PER_NOTE = 2  # drills retrieved per coaching note

_SYSTEM_PROMPT = """\
You are a tennis coach writing a personalised 1-week serve training plan.

You have been given:
  1. Coaching notes from a biomechanical serve analysis, with priority levels
  2. A set of drills retrieved from a curated knowledge base

Rules:
- Use ONLY the provided drills. Do not invent new drills or reference ones not given.
- Weight the schedule toward high-priority fixes — these should appear most days.
- Medium/low priority items are maintenance; include them 1–2 times across the week.
- Structure: Day 1–5 (Monday–Friday). Each day: 2–3 drills with sets and reps.
  Day 6 (Saturday): optional match-play applying the week's focus.
  Day 7 (Sunday): rest.
- Reference the player's actual measured numbers when relevant (e.g., "your knee angle
  was 169° — we're targeting 140°").
- Tone: encouraging, specific, and actionable. Explain briefly WHY each drill is in
  that day's session.\
"""


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


def _load_drills() -> list[dict]:
    with open(_DRILLS_PATH) as f:
        return json.load(f)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def retrieve_drills(feedback: dict) -> list[dict]:
    """Return deduplicated list of most relevant drills for the given feedback."""
    drills = _load_drills()
    print(f"Loading embedding model ({_EMBED_MODEL}) …")
    model = SentenceTransformer(_EMBED_MODEL)

    # Embed all drill descriptions
    drill_texts = [d["description"] for d in drills]
    drill_embeddings = model.encode(drill_texts, convert_to_numpy=True)

    # Build one query per coaching note
    all_notes = []
    for level in ("high", "medium", "low"):
        all_notes.extend(feedback.get(level, []))

    seen_ids: set[str] = set()
    retrieved: list[dict] = []

    for note in all_notes:
        query = f"{note['metric']}: {note['observation']}"
        query_embedding = model.encode([query], convert_to_numpy=True)
        sims = _cosine_similarity(query_embedding, drill_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:_TOP_K_PER_NOTE]
        for idx in top_indices:
            drill = drills[idx]
            if drill["id"] not in seen_ids:
                seen_ids.add(drill["id"])
                retrieved.append({**drill, "_score": float(sims[idx])})

    # Sort by score descending
    retrieved.sort(key=lambda d: d["_score"], reverse=True)
    return retrieved


def _build_user_message(feedback: dict, retrieved_drills: list[dict]) -> str:
    lines = ["=== COACHING NOTES ===\n"]

    for level in ("high", "medium", "low"):
        notes = feedback.get(level, [])
        if not notes:
            continue
        label = {"high": "HIGH PRIORITY (fix these first)",
                 "medium": "MEDIUM PRIORITY (room to improve)",
                 "low": "LOW PRIORITY (doing well / informational)"}[level]
        lines.append(f"[{label}]")
        for n in notes:
            lines.append(f"  Metric: {n['metric']}")
            lines.append(f"  Observation: {n['observation']}")
            lines.append(f"  Recommendation: {n['recommendation']}")
            lines.append("")

    lines.append("\n=== RETRIEVED DRILLS ===\n")
    for d in retrieved_drills:
        lines.append(f"Drill: {d['title']}  (id: {d['id']})")
        lines.append(f"Focus: {d['focus_metric']}")
        lines.append(f"Description: {d['description']}")
        lines.append("")

    lines.append("\nWrite a 1-week (Mon–Sun) training plan using only the drills above.")
    return "\n".join(lines)


def generate_plan(feedback: dict, retrieved_drills: list[dict]) -> str:
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(feedback, retrieved_drills)}],
    )
    return response.content[0].text


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python generate_training_plan.py <feedback.json>")

    feedback_path = Path(sys.argv[1])
    if not feedback_path.exists():
        sys.exit(f"File not found: {feedback_path}")

    with open(feedback_path) as f:
        feedback = json.load(f)

    print(f"Loaded feedback from {feedback_path.name}")

    # Step 1: Retrieve relevant drills
    retrieved = retrieve_drills(feedback)
    print(f"Retrieved {len(retrieved)} drills from knowledge base:\n")
    for d in retrieved:
        print(f"  [{d['_score']:.3f}] {d['title']}  ({d['focus_metric']})")

    # Step 2: Generate plan
    print("\nGenerating training plan …\n")
    plan_text = generate_plan(feedback, retrieved)

    print("=== 1-WEEK TRAINING PLAN ===\n")
    print(plan_text)

    # Step 3: Save outputs
    stem = feedback_path.name.replace("_feedback.json", "")
    out_dir = feedback_path.parent

    txt_path = out_dir / f"{stem}_training_plan.txt"
    txt_path.write_text(plan_text)
    print(f"\nSaved → {txt_path}")

    json_path = out_dir / f"{stem}_training_plan.json"
    with open(json_path, "w") as f:
        json.dump({"retrieved_drills": retrieved, "plan": plan_text}, f, indent=2)
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()
