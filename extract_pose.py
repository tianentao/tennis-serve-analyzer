import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

_vision = mp.tasks.vision
_drawing = _vision.drawing_utils
_drawing_styles = _vision.drawing_styles
_PoseLandmarker = _vision.PoseLandmarker
_PoseLandmarkerOptions = _vision.PoseLandmarkerOptions
_BaseOptions = mp.tasks.BaseOptions
_RunningMode = _vision.RunningMode
_CONNECTIONS = list(_vision.PoseLandmarksConnections.POSE_LANDMARKS)

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODEL_NAMES = {0: "pose_landmarker_lite.task", 1: "pose_landmarker_full.task", 2: "pose_landmarker_heavy.task"}


def _ensure_model(complexity: int) -> str:
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / _MODEL_NAMES[complexity]
    if not path.exists():
        url = _MODEL_URLS[complexity]
        print(f"Downloading model ({_MODEL_NAMES[complexity]}) …")
        urllib.request.urlretrieve(url, path)
        print(f"  saved → {path}")
    return str(path)


def extract_pose(input_path: str, output_path: str, model_complexity: int = 1) -> None:
    model_path = _ensure_model(model_complexity)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    snap_indices = {0, total_frames // 2, max(0, total_frames - 1)}
    stem = Path(output_path).stem
    snap_labels = {0: "first", total_frames // 2: "mid", max(0, total_frames - 1): "last"}

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmark_style = _drawing_styles.get_default_pose_landmarks_style()

    with _PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                _drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks[0],
                    _CONNECTIONS,
                    landmark_drawing_spec=landmark_style,
                )

            out.write(frame)

            if frame_idx in snap_indices:
                label = snap_labels[frame_idx]
                snap_path = str(Path(output_path).parent / f"{stem}_{label}.jpg")
                cv2.imwrite(snap_path, frame)
                print(f"  snapshot → {snap_path}")

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  processed {frame_idx}/{total_frames} frames", end="\r")

    cap.release()
    out.release()
    print(f"\nDone. Overlay video → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay MediaPipe pose landmarks on a serve video.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--out", help="Output video path (default: <input>_pose.mp4)")
    parser.add_argument(
        "--complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Model complexity: 0=lite/fast, 1=full/balanced (default), 2=heavy/accurate",
    )
    args = parser.parse_args()

    input_path = args.input_video
    output_path = args.out or str(
        Path(input_path).with_stem(Path(input_path).stem + "_pose").with_suffix(".mp4")
    )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  complexity={args.complexity} ({_MODEL_NAMES[args.complexity]})")
    extract_pose(input_path, output_path, model_complexity=args.complexity)


if __name__ == "__main__":
    main()
