"""Evaluate frame predictions and per-clip probability aggregation on held-out videos."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from calibration import calibrated_probabilities, load_temperature
from dataset import CLASSES
from infer_video import frame_to_tensor
from metrics import compute_metrics
from model import build_model_from_config
from provenance import sha256_file


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


def collect_video_files(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"Video dataset directory does not exist: {root}")
    found_classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if found_classes != sorted(CLASSES):
        raise ValueError(f"Expected exactly the eight class folders {CLASSES}; found {found_classes}")
    videos = []
    for label, class_name in enumerate(CLASSES):
        class_videos = sorted(path for path in (root / class_name).rglob("*")
                              if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
        if not class_videos:
            raise ValueError(f"No supported video files found under class folder '{class_name}'")
        videos.extend((path, label) for path in class_videos)
    return videos


def evaluate_one_video(cv2, model, path: Path, label: int, class_names, device,
                       temperature: float, sample_stride: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {path}")
    frame_probs, frame_predictions = [], []
    decoded_frames = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames += 1
                if (decoded_frames - 1) % sample_stride:
                    continue
                logits = model(frame_to_tensor(frame, device))
                probabilities = calibrated_probabilities(logits, temperature)[0].cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise RuntimeError(f"Non-finite probabilities for {path} at decoded frame {decoded_frames}")
                frame_probs.append(probabilities)
                frame_predictions.append(int(np.argmax(probabilities)))
    finally:
        capture.release()
    elapsed = max(time.perf_counter() - started, 1e-9)
    if not frame_probs:
        raise RuntimeError(f"No frames were evaluated in video: {path}")
    probability_matrix = np.asarray(frame_probs, dtype=np.float64)
    video_probabilities = probability_matrix.mean(axis=0)
    transitions = max(len(frame_predictions) - 1, 0)
    switch_count = sum(a != b for a, b in zip(frame_predictions, frame_predictions[1:]))
    return {
        "label": label,
        "frame_probabilities": probability_matrix,
        "frame_predictions": frame_predictions,
        "video_probabilities": video_probabilities,
        "decoded_frames": decoded_frames,
        "evaluated_frames": len(frame_probs),
        "class_switches": switch_count,
        "class_switch_rate": switch_count / transitions if transitions else 0.0,
        "processing_fps": len(frame_probs) / elapsed,
    }


def main(args):
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be at least 1")
    if not args.confirm_independent:
        raise ValueError("Pass --confirm-independent only after confirming this is an untouched video cohort")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Video evaluation requires OpenCV. Install it with `python -m pip install opencv-python`.") from exc

    video_root = Path(args.data_root).expanduser().resolve()
    videos = collect_video_files(video_root)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    class_names = config.get("class_names", CLASSES)
    if class_names != CLASSES or config.get("num_classes") != len(CLASSES):
        raise ValueError("Video folder labels must match the checkpoint's eight Kvasir classes and order")
    temperature, calibration = load_temperature(args.calibration, checkpoint_path, class_names)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels_per_video, video_probabilities, frame_labels, frame_probabilities = [], [], [], []
    records = []
    evaluation_started = time.perf_counter()
    for index, (video_path, label) in enumerate(videos, start=1):
        result = evaluate_one_video(cv2, model, video_path, label, class_names, device,
                                    temperature, args.sample_stride)
        relative_path = video_path.relative_to(video_root).as_posix()
        labels_per_video.append(label)
        video_probabilities.append(result["video_probabilities"])
        frame_labels.extend([label] * result["evaluated_frames"])
        frame_probabilities.extend(result["frame_probabilities"].tolist())
        ranking = np.argsort(result["video_probabilities"])[::-1]
        record = {
            "relative_video_path": relative_path,
            "video_sha256": sha256_file(video_path),
            "true_class": class_names[label],
            "predicted_class": class_names[int(ranking[0])],
            "top_probability": float(result["video_probabilities"][ranking[0]]),
            "evaluated_frames": result["evaluated_frames"],
            "decoded_frames": result["decoded_frames"],
            "class_switches": result["class_switches"],
            "class_switch_rate": result["class_switch_rate"],
            "processing_fps": result["processing_fps"],
            "class_probabilities_json": json.dumps(
                {name: float(result["video_probabilities"][i]) for i, name in enumerate(class_names)},
                separators=(",", ":"), allow_nan=False),
        }
        records.append(record)
        print(f"[{index}/{len(videos)}] {relative_path}: {record['predicted_class']} "
              f"({record['top_probability']:.3f}), frames={record['evaluated_frames']}, "
              f"switch_rate={record['class_switch_rate']:.3f}")

    frame_metrics = compute_metrics(frame_labels, frame_probabilities, class_names)
    video_metrics = compute_metrics(labels_per_video, video_probabilities, class_names)
    total_frames = sum(record["evaluated_frames"] for record in records)
    elapsed = max(time.perf_counter() - evaluation_started, 1e-9)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_name(f"{output.stem}_per_video.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "evaluation_type": "external held-out labeled videos; per-frame classifier with mean-probability clip aggregation",
        "dataset_id": args.dataset_id,
        "video_root": str(video_root),
        "video_count": len(records),
        "evaluated_frame_count": total_frames,
        "sample_stride": args.sample_stride,
        "model_checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "class_names": class_names,
        "device": str(device),
        "confidence_calibration": "validation_temperature_scaled" if calibration else "raw_uncalibrated_softmax",
        "temperature": temperature if calibration else None,
        "frame_metrics": frame_metrics,
        "video_aggregated_metrics": video_metrics,
        "mean_per_video_class_switch_rate": float(np.mean([row["class_switch_rate"] for row in records])),
        "mean_per_video_processing_fps": float(np.mean([row["processing_fps"] for row in records])),
        "overall_evaluation_fps_including_video_decode_and_hashing": total_frames / elapsed,
        "operator_attestation": {"independent_video_cohort_confirmed": True,
                                 "provided_by_command_flag": "--confirm-independent"},
        "limitations": [
            "This script does not add temporal reasoning; each sampled frame is classified independently.",
            "Video-level scores are the mean of sampled frame probabilities and assume one ground-truth class per clip.",
            "The operator attestation does not prove patient/procedure independence or label quality.",
            "Class-switch rate measures prediction instability, not lesion tracking or clinical safety.",
        ],
        "per_video_csv": str(csv_path),
        "per_video_csv_sha256": sha256_file(csv_path),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved video evaluation to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a frame classifier on an independent labeled video cohort")
    parser.add_argument("--data-root", required=True,
                        help="Eight class folders, each containing one or more labeled video files")
    parser.add_argument("--dataset-id", required=True, help="Source/version/site/date identifier for the video cohort")
    parser.add_argument("--confirm-independent", action="store_true",
                        help="Attest that videos are untouched and independent of model development data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default=None,
                        help="Optional validation-fitted calibration JSON for this exact checkpoint")
    parser.add_argument("--sample-stride", type=int, default=5,
                        help="Evaluate every Nth decoded frame (default 5)")
    parser.add_argument("--output", default="video_metrics.json")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
