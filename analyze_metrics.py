"""
Phase 1 metric extraction: joint angles and biomechanical ratios at the four
key serve frames. Imports key-frame detection from detect_contact.py.

Metrics
-------
  contact:
    elbow_extension      — racket elbow angle at impact (target 160–175°)
    contact_height_ratio — wrist height / body height (target ≥ 0.90)
  trophy:
    knee_bend            — racket-side knee angle (target 130–155°; lower = more bent)
  toss_zenith:
    toss_height_ratio    — toss wrist height above shoulder / body height (target ≥ 0.08)
  drop:
    drop_elbow_angle     — racket elbow at backscratch (informational)

Output: JSON printed to stdout + saved as <stem>_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from detect_contact import _ensure_pose_model, detect_key_frames, find_contact_frame

# MediaPipe pose landmark indices
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW,    _R_ELBOW    = 13, 14
_L_WRIST,    _R_WRIST    = 15, 16
_L_HIP,      _R_HIP      = 23, 24
_L_KNEE,     _R_KNEE     = 25, 26
_L_ANKLE,    _R_ANKLE    = 27, 28


def _angle_deg(a, b, c) -> float:
    """Angle in degrees at vertex b between rays b→a and b→c."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _pt(lms, idx, w, h) -> tuple[float, float]:
    lm = lms[idx]
    return (lm.x * w, lm.y * h)


def _body_height_px(lms, w, h) -> float:
    """Approximate body height in pixels: avg-ankle y − nose y."""
    nose_y = _pt(lms, _NOSE, w, h)[1]
    ankle_y = (_pt(lms, _L_ANKLE, w, h)[1] + _pt(lms, _R_ANKLE, w, h)[1]) / 2
    return ankle_y - nose_y  # positive: ankles are below nose in frame


def _metric(value: float, unit: str, lo: float | None = None, hi: float | None = None,
            flag: str | None = None) -> dict:
    if flag is None:
        if lo is not None and value < lo:
            flag = f"low — target {lo}–{hi} {unit}"
        elif hi is not None and value > hi:
            flag = f"high — target {lo}–{hi} {unit}"
    return {"value": round(value, 2), "unit": unit, "flag": flag}


def _load_landmarks_batch(video_path: str, frame_indices: list[int]) -> tuple[dict, int, int]:
    """Return ({frame_idx: landmark_list | None}, width, height)."""
    model_path = _ensure_pose_model()
    _vision = mp.tasks.vision
    options = _vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results: dict = {}
    with _vision.PoseLandmarker.create_from_options(options) as landmarker:
        for fi in sorted(set(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                results[fi] = None
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            results[fi] = r.pose_landmarks[0] if r.pose_landmarks else None
    cap.release()
    return results, w, h


def _serving_arm_indices(lms, w, h) -> tuple[tuple, tuple]:
    """
    Return (racket_arm, toss_arm) index tuples: (shoulder, elbow, wrist, hip, knee, ankle).
    The racket wrist is the one that is higher (smaller y) at contact.
    """
    lw_y = _pt(lms, _L_WRIST, w, h)[1]
    rw_y = _pt(lms, _R_WRIST, w, h)[1]
    if lw_y < rw_y:  # left wrist higher → left-handed
        racket = (_L_SHOULDER, _L_ELBOW, _L_WRIST, _L_HIP, _L_KNEE, _L_ANKLE)
        toss   = (_R_SHOULDER, _R_ELBOW, _R_WRIST, _R_HIP, _R_KNEE, _R_ANKLE)
    else:             # right wrist higher → right-handed
        racket = (_R_SHOULDER, _R_ELBOW, _R_WRIST, _R_HIP, _R_KNEE, _R_ANKLE)
        toss   = (_L_SHOULDER, _L_ELBOW, _L_WRIST, _L_HIP, _L_KNEE, _L_ANKLE)
    return racket, toss


def analyze_metrics(video_path: str, key_frames: dict[str, int]) -> dict:
    lm_map, w, h = _load_landmarks_batch(video_path, list(key_frames.values()))

    contact_lms = lm_map.get(key_frames["contact"])
    if contact_lms is None:
        sys.exit("No pose detected at contact frame — can't determine serving arm.")

    racket, toss = _serving_arm_indices(contact_lms, w, h)
    r_sh, r_el, r_wr, r_hip, r_knee, r_ank = racket
    t_sh, t_el, t_wr, t_hip, t_knee, t_ank = toss

    metrics: dict[str, dict] = {}

    # ── Contact frame ──────────────────────────────────────────────────────────
    lms = contact_lms
    bh = _body_height_px(lms, w, h)

    elbow_ext = _angle_deg(_pt(lms, r_sh, w, h), _pt(lms, r_el, w, h), _pt(lms, r_wr, w, h))
    # Upper bound omitted: full extension (180°) is ideal, not a flaw
    metrics["elbow_extension"] = _metric(elbow_ext, "°", lo=160)

    # (ankle_y − wrist_y) / body_height: how far above the ankles is the racket wrist
    ankle_y = (_pt(lms, _L_ANKLE, w, h)[1] + _pt(lms, _R_ANKLE, w, h)[1]) / 2
    wrist_y = _pt(lms, r_wr, w, h)[1]
    height_ratio = (ankle_y - wrist_y) / bh
    flag = "low contact point — try to connect higher" if height_ratio < 0.90 else None
    metrics["contact_height_ratio"] = _metric(height_ratio, "body_ratio", flag=flag)

    # ── Trophy frame ───────────────────────────────────────────────────────────
    lms = lm_map.get(key_frames["trophy"])
    if lms:
        knee_angle = _angle_deg(
            _pt(lms, r_hip, w, h),
            _pt(lms, r_knee, w, h),
            _pt(lms, r_ank, w, h),
        )
        flag = None
        if knee_angle > 165:
            flag = "legs too straight — bend knees to load more power"
        elif knee_angle < 120:
            flag = "very deep bend — may limit upward drive"
        metrics["knee_bend"] = _metric(knee_angle, "°", lo=130, hi=155, flag=flag)

    # ── Toss zenith frame ──────────────────────────────────────────────────────
    lms = lm_map.get(key_frames["toss_zenith"])
    if lms:
        bh_tz = _body_height_px(lms, w, h)
        shoulder_y = _pt(lms, t_sh, w, h)[1]
        toss_wrist_y = _pt(lms, t_wr, w, h)[1]
        # Positive when wrist is above shoulder (smaller y = higher in frame)
        toss_ratio = (shoulder_y - toss_wrist_y) / bh_tz
        flag = "toss barely above shoulder — toss higher for more time" if toss_ratio < 0.05 else None
        metrics["toss_height_ratio"] = _metric(toss_ratio, "body_ratio", lo=0.08, hi=None, flag=flag)

    # ── Racket drop frame (informational) ──────────────────────────────────────
    lms = lm_map.get(key_frames["drop"])
    if lms:
        drop_angle = _angle_deg(
            _pt(lms, r_sh, w, h),
            _pt(lms, r_el, w, h),
            _pt(lms, r_wr, w, h),
        )
        metrics["drop_elbow_angle"] = _metric(drop_angle, "°")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract serve biomechanics metrics.")
    parser.add_argument("input_video")
    args = parser.parse_args()

    print(f"Detecting key frames: {args.input_video}")
    candidate_frame, toss_zenith_frame, track = find_contact_frame(args.input_video)
    candidate_det = next(((cx, cy) for fi, cx, cy in track if fi == candidate_frame), None)
    ball_cy = candidate_det[1] if candidate_det else 0

    key_frames = detect_key_frames(args.input_video, candidate_frame, ball_cy)
    key_frames["toss_zenith"] = toss_zenith_frame

    print(f"\nKey frames: {key_frames}")
    print("Extracting metrics …\n")

    metrics = analyze_metrics(args.input_video, key_frames)

    print("=== Serve Metrics ===")
    for name, m in metrics.items():
        flag_str = f"  ← {m['flag']}" if m["flag"] else ""
        print(f"  {name:25s}  {m['value']:6.1f} {m['unit']}{flag_str}")

    out_path = Path(args.input_video).parent / (Path(args.input_video).stem + "_metrics.json")
    payload = {"key_frames": key_frames, "metrics": metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
