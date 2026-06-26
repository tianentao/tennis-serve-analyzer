"""
Detects three key frames from a tennis serve video:
  1. Trophy position  — toss arm at peak, racket arm cocked
  2. Racket drop      — racket at its lowest point (backscratch)
  3. Contact          — racket meets ball

Strategy:
  - HSV ball detection finds the toss arc → candidate frame
  - PoseLandmarker wrist trajectories derive all three key frames
    from a single scan window around the candidate

Output:
  - <stem>_trophy.jpg, <stem>_drop.jpg, <stem>_contact.jpg
  - <stem>_ball_track.jpg — full arc visualization for debugging
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Pose model (full = complexity 1) for wrist-based contact refinement
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_POSE_MODEL_NAME = "pose_landmarker_full.task"


def _ensure_pose_model() -> str:
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / _POSE_MODEL_NAME
    if not path.exists():
        print(f"Downloading pose model …")
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
    return str(path)

# HSV range calibrated from ball pixels in this video.
# Ball H=37-40, S=100-128, V=100-142. Keeping S/V mins at 80 to tolerate
# motion blur (which desaturates the ball near the top of the toss).
# H upper at 58 to reject cyan/blue noise at H=64.
_HSV_LO = np.array([28, 80, 85])
_HSV_HI = np.array([58, 255, 255])
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Blob filters
_MIN_AREA = 100
_MAX_AREA = 2500
_MIN_CIRC = 0.50


def _detect_ball(frame: np.ndarray, max_y: int) -> tuple[int, int, float] | None:
    """Return (cx, cy, circularity) of the best ball candidate, or None."""
    h_frame, w_frame = frame.shape[:2]
    x_margin = int(w_frame * 0.04)  # ignore blobs within 4% of left/right edge

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_AREA or area > _MAX_AREA:
            continue
        perimeter = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        if circ < _MIN_CIRC:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        if cy > max_y or cx < x_margin or cx > w_frame - x_margin:
            continue
        if best is None or circ > best[2]:
            best = (cx, cy, circ)

    return best


def find_contact_frame(video_path: str) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Returns (contact_frame_index, ball_track).
    ball_track is a list of (frame_idx, cx, cy) detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Restrict to upper 42% of frame. Court markings (the main false positive) live
    # at y=840-1070; ball during toss tops out at y=787. This cuts all court noise.
    max_y = int(height * 0.42)

    track: list[tuple[int, int, int]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = _detect_ball(frame, max_y)
        if result:
            cx, cy, _ = result
            track.append((frame_idx, cx, cy))
        frame_idx += 1

    cap.release()

    if not track:
        sys.exit("No ball detections found. Check HSV range or video quality.")

    # Group detections into runs: consecutive detections with frame gap ≤ 5.
    # A run represents one continuous period where the ball is visible.
    _MAX_FRAME_GAP = 5
    _MIN_RUN_LEN = 3
    runs: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = [track[0]]
    for det in track[1:]:
        if det[0] - current[-1][0] <= _MAX_FRAME_GAP:
            current.append(det)
        else:
            if len(current) >= _MIN_RUN_LEN:
                runs.append(current)
            current = [det]
    if len(current) >= _MIN_RUN_LEN:
        runs.append(current)

    if not runs:
        print("Warning: no sustained ball runs found; using raw track minimum-y.")
        contact_frame = min(track, key=lambda t: t[2])[0]
        return contact_frame, track

    # The toss arc is the run that reaches highest (smallest y value).
    # All other runs (ball in hand, ball on court) stay at mid-to-low y.
    toss_run = min(runs, key=lambda run: min(cy for _, _, cy in run))
    toss_zenith_frame, _, toss_peak_y = min(toss_run, key=lambda d: d[2])
    print(f"Toss arc: frames {toss_run[0][0]}–{toss_run[-1][0]}, peak y={toss_peak_y} (frame {toss_zenith_frame})")

    # Contact = last frame where ball is detected in the toss arc.
    contact_frame = toss_run[-1][0]
    return contact_frame, toss_zenith_frame, track


def detect_key_frames(video_path: str, candidate_frame: int, ball_cy: int) -> dict[str, int]:
    """
    Derive trophy, drop, and contact frame indices from wrist landmark trajectories.

    Scans [candidate_frame-5 .. candidate_frame+42] with PoseLandmarker (IMAGE mode).
    From per-frame LEFT_WRIST and RIGHT_WRIST y positions:
      - Trophy  = frame where LEFT_WRIST (toss arm) is highest (min y)
      - Drop    = frame where RIGHT_WRIST (racket arm) is lowest (max y) between trophy and contact
      - Contact = frame after RIGHT_WRIST reaches its highest point (min y)
    """
    model_path = _ensure_pose_model()
    _vision = mp.tasks.vision

    options = _vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = max(0, candidate_frame - 5)
    end = candidate_frame + 42
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Collect (frame_idx, left_wrist_y, right_wrist_y) for every detected frame
    wrist_data: list[tuple[int, int, int]] = []

    with _vision.PoseLandmarker.create_from_options(options) as landmarker:
        for fi in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.pose_landmarks:
                continue
            lms = result.pose_landmarks[0]
            lw_y = int(lms[15].y * height)  # LEFT_WRIST  = toss arm (right-handed player)
            rw_y = int(lms[16].y * height)  # RIGHT_WRIST = racket arm
            wrist_data.append((fi, lw_y, rw_y))

    cap.release()

    if not wrist_data:
        return {"trophy": candidate_frame, "drop": candidate_frame, "contact": candidate_frame + 1}

    # Trophy: toss arm (LEFT_WRIST) at its highest point
    trophy_frame = min(wrist_data, key=lambda d: d[1])[0]

    # Racket arm (RIGHT_WRIST) peak: highest point after trophy = just before contact
    post_trophy = [d for d in wrist_data if d[0] >= trophy_frame]
    rw_peak_frame = min(post_trophy, key=lambda d: d[2])[0]

    # Drop: racket arm at its lowest between trophy and the peak swing
    between = [d for d in wrist_data if trophy_frame <= d[0] <= rw_peak_frame]
    drop_frame = max(between, key=lambda d: d[2])[0] if between else trophy_frame

    # Contact: 1 frame after the racket arm peaks (empirically validated)
    contact_frame = rw_peak_frame + 1

    print(f"Trophy frame:  {trophy_frame}  (t={trophy_frame/30:.2f}s)")
    print(f"Racket drop:   {drop_frame}  (t={drop_frame/30:.2f}s)")
    print(f"Contact frame: {contact_frame}  (t={contact_frame/30:.2f}s)")

    return {"trophy": trophy_frame, "drop": drop_frame, "contact": contact_frame}


def save_outputs(
    video_path: str,
    key_frames: dict[str, int],
    track: list[tuple[int, int, int]],
    ball_pos: tuple[int, int] | None = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    stem = Path(video_path).stem
    out_dir = Path(video_path).parent

    labels = {
        "toss_zenith": ("TOSS ZENITH", (255, 255, 0)),  # yellow
        "trophy": ("TROPHY", (255, 165, 0)),            # orange
        "drop":   ("RACKET DROP", (0, 165, 255)),       # amber
        "contact": ("CONTACT", (0, 255, 0)),            # green
    }

    for key, (label, color) in labels.items():
        fi = key_frames[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        # Annotate ball position on contact frame only
        if key == "contact" and ball_pos:
            cv2.circle(frame, ball_pos, 30, color, 4)
        cv2.putText(frame, label, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
        cv2.putText(frame, f"frame {fi}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
        out_path = str(out_dir / f"{stem}_{key}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  {label} → {out_path}")

    # --- Ball track visualization ---
    contact_fi = key_frames["contact"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, contact_fi)
    ok, base = cap.read()
    if ok:
        for fi, cx, cy in track:
            color = (0, 255, 0) if fi == contact_fi else (255, 165, 0)
            radius = 14 if fi == contact_fi else 8
            cv2.circle(base, (cx, cy), radius, color, -1)
        track_path = str(out_dir / f"{stem}_ball_track.jpg")
        cv2.imwrite(track_path, base)
        print(f"  Ball track → {track_path}")

    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the ball-contact frame in a tennis serve video.")
    parser.add_argument("input_video")
    args = parser.parse_args()

    print(f"Scanning: {args.input_video}")
    candidate_frame, toss_zenith_frame, track = find_contact_frame(args.input_video)
    print(f"Ball candidate frame: {candidate_frame}  (t={candidate_frame/30:.2f}s)")
    print(f"Ball detections: {len(track)} frames\n")

    candidate_det = next(((cx, cy) for fi, cx, cy in track if fi == candidate_frame), None)
    ball_cy = candidate_det[1] if candidate_det else 0

    key_frames = detect_key_frames(args.input_video, candidate_frame, ball_cy)
    key_frames["toss_zenith"] = toss_zenith_frame
    print()
    save_outputs(args.input_video, key_frames, track, ball_pos=candidate_det)


if __name__ == "__main__":
    main()
