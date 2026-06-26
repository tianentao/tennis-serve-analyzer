"""
MCP client that chains the three tennis analysis tools in sequence:
  analyze_serve → generate_feedback → generate_training_plan

Usage:
    python tennis_mcp_client.py <video_path>

The client launches the MCP server as a subprocess (stdio transport),
calls each tool in order, and prints a summary of results.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_pipeline(video_path: str) -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "tennis_mcp_server.py")],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Connected to MCP server. Tools available: {[t.name for t in tools.tools]}\n")

            # ── Step 1: analyze_serve ──────────────────────────────────────
            print("=" * 60)
            print("STEP 1 — Analyzing serve video …")
            print("=" * 60)
            result = await session.call_tool("analyze_serve", {"video_path": video_path})
            metrics_json = result.content[0].text
            payload = json.loads(metrics_json)

            print(f"Key frames detected: {payload['key_frames']}")
            print("\nMetrics:")
            for name, m in payload["metrics"].items():
                flag_str = f"  ← {m['flag']}" if m["flag"] else ""
                print(f"  {name:25s}  {m['value']:6.1f} {m['unit']}{flag_str}")

            # ── Step 2: generate_feedback ──────────────────────────────────
            print("\n" + "=" * 60)
            print("STEP 2 — Generating coaching feedback …")
            print("=" * 60)
            result = await session.call_tool("generate_feedback", {"metrics_json": metrics_json})
            feedback_json = result.content[0].text
            feedback = json.loads(feedback_json)

            for level in ("high", "medium", "low"):
                notes = feedback.get(level, [])
                if not notes:
                    continue
                label = {"high": "PRIORITY FIXES", "medium": "ROOM TO IMPROVE", "low": "DOING WELL"}[level]
                print(f"\n[{label}]")
                for n in notes:
                    print(f"  {n['metric']}: {n['recommendation']}")

            # ── Step 3: generate_training_plan ─────────────────────────────
            print("\n" + "=" * 60)
            print("STEP 3 — Generating 1-week training plan …")
            print("=" * 60)
            result = await session.call_tool("generate_training_plan", {"feedback_json": feedback_json})
            plan_payload = json.loads(result.content[0].text)

            print(f"\nRetrieved {len(plan_payload['retrieved_drills'])} drills from knowledge base.")
            print("\n--- TRAINING PLAN ---\n")
            print(plan_payload["plan"])

            # Save all outputs next to the video
            stem = Path(video_path).stem
            out_dir = Path(video_path).parent
            (out_dir / f"{stem}_metrics.json").write_text(metrics_json)
            (out_dir / f"{stem}_feedback.json").write_text(feedback_json)
            (out_dir / f"{stem}_training_plan.txt").write_text(plan_payload["plan"])
            (out_dir / f"{stem}_training_plan.json").write_text(
                json.dumps(plan_payload, indent=2)
            )
            print(f"\nAll outputs saved to {out_dir}/")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python tennis_mcp_client.py <video_path>")
    asyncio.run(run_pipeline(sys.argv[1]))


if __name__ == "__main__":
    main()
