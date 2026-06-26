"""
MCP server for the tennis serve analysis pipeline.

Exposes three tools that chain together:
  1. analyze_serve(video_path)       → metrics JSON
  2. generate_feedback(metrics_json) → feedback JSON
  3. generate_training_plan(feedback_json) → training plan text

Run via stdio (standard MCP transport):
    python tennis_mcp_server.py
"""

import asyncio
import contextlib
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Ensure the project root is on the path so sibling modules import cleanly
sys.path.insert(0, str(Path(__file__).parent))

import detect_contact as _dc
import analyze_metrics as _am
import generate_feedback as _gf
import generate_training_plan as _gtp

server = Server("tennis-serve-analyzer")


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect stdout to stderr while pipeline code runs.
    MCP uses stdout for JSON-RPC; any other text there corrupts the channel."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_serve",
            description=(
                "Run Phase 1 + 2 pipeline on a tennis serve video: detect key frames "
                "(trophy, drop, contact, toss zenith), extract biomechanical metrics "
                "(elbow extension, knee bend, contact height, toss height, drop elbow), "
                "and return structured JSON findings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the serve video file.",
                    }
                },
                "required": ["video_path"],
            },
        ),
        Tool(
            name="generate_feedback",
            description=(
                "Run Phase 2: take metrics JSON produced by analyze_serve and call "
                "Claude to generate encouraging, specific coaching notes for each "
                "metric, grouped by priority (high / medium / low)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "metrics_json": {
                        "type": "string",
                        "description": "JSON string as returned by analyze_serve.",
                    }
                },
                "required": ["metrics_json"],
            },
        ),
        Tool(
            name="generate_training_plan",
            description=(
                "Run Phase 3: take feedback JSON produced by generate_feedback, "
                "retrieve relevant drills from the knowledge base using vector "
                "similarity, and generate a personalised 1-week training plan "
                "grounded in those drills."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "feedback_json": {
                        "type": "string",
                        "description": "JSON string as returned by generate_feedback.",
                    }
                },
                "required": ["feedback_json"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    with _stdout_to_stderr():
        if name == "analyze_serve":
            video_path = arguments["video_path"]

            candidate_frame, toss_zenith_frame, track = _dc.find_contact_frame(video_path)
            candidate_det = next(
                ((cx, cy) for fi, cx, cy in track if fi == candidate_frame), None
            )
            ball_cy = candidate_det[1] if candidate_det else 0
            key_frames = _dc.detect_key_frames(video_path, candidate_frame, ball_cy)
            key_frames["toss_zenith"] = toss_zenith_frame

            metrics = _am.analyze_metrics(video_path, key_frames)
            payload = {"key_frames": key_frames, "metrics": metrics}
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        elif name == "generate_feedback":
            payload = json.loads(arguments["metrics_json"])
            metrics = payload["metrics"]
            notes = _gf.generate_feedback(metrics)

            grouped: dict[str, list] = {"high": [], "medium": [], "low": []}
            for note in notes:
                grouped.setdefault(note["priority"], []).append(note)
            return [TextContent(type="text", text=json.dumps(grouped, indent=2))]

        elif name == "generate_training_plan":
            feedback = json.loads(arguments["feedback_json"])
            retrieved = _gtp.retrieve_drills(feedback)
            plan_text = _gtp.generate_plan(feedback, retrieved)

            payload = {"retrieved_drills": retrieved, "plan": plan_text}
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
