"""Run frame-by-frame classification on a webcam, capture card, or video file."""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from calibration import calibrated_probabilities, load_temperature
from dataset import CLASSES
from infer import largest_component_box, make_cam, map_is_usable, top_region_mask
from model import build_model_from_config


def resolve_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def frame_to_tensor(frame_bgr: np.ndarray, device):
    frame_rgb = frame_bgr[..., ::-1]
    image = Image.fromarray(np.ascontiguousarray(frame_rgb))
    image = ImageOps.fit(image, (224, 224), method=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def explain_frame(model, frame_bgr: np.ndarray, device, class_names, temperature: float):
    """Make Grad-CAM/SpikeCAM evidence for a single frame; this costs extra inference time."""
    with torch.inference_mode(False), torch.enable_grad():
        image = frame_to_tensor(frame_bgr, device).requires_grad_(True)
        logits, aux = model.forward_with_aux(image)
        probabilities = calibrated_probabilities(logits, temperature)[0].detach().cpu().numpy()
        target = int(np.argmax(probabilities))
        gradcam = make_cam(logits[0, target], aux["cnn_map"], retain_graph=True)
        spikecam = make_cam(logits[0, target], aux["snn_map"])

    available = [cam for cam in (gradcam, spikecam) if map_is_usable(cam)]
    combined = np.mean(available, axis=0) if available else np.zeros((224, 224), dtype=np.float32)
    box = largest_component_box(top_region_mask(combined)) if available else None
    score = float(probabilities[target])
    if box is not None:
        text = (f"Model evidence favors {class_names[target]} ({score:.1%}); red box marks an attribution region, "
                "not a confirmed lesion boundary.")
    else:
        text = (f"Model evidence favors {class_names[target]} ({score:.1%}); no stable attribution box was available.")
    return target, score, combined, box, text


def draw_explanation_overlay(frame, heatmap, box, text, cv2):
    """Map the 224-square center crop back onto the corresponding region of the source frame."""
    height, width = frame.shape[:2]
    crop_side = min(height, width)
    crop_x = (width - crop_side) // 2
    crop_y = (height - crop_side) // 2
    crop = frame[crop_y:crop_y + crop_side, crop_x:crop_x + crop_side]
    if np.isfinite(heatmap).all() and float(np.ptp(heatmap)) > 1e-8:
        heat = cv2.resize((np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
                          (crop_side, crop_side), interpolation=cv2.INTER_LINEAR)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        frame[crop_y:crop_y + crop_side, crop_x:crop_x + crop_side] = cv2.addWeighted(
            crop, 0.65, heat_color, 0.35, 0.0)
    if box is not None:
        x1, y1, x2, y2 = box
        left = crop_x + int(x1 * crop_side / 224)
        top = crop_y + int(y1 * crop_side / 224)
        right = crop_x + int((x2 + 1) * crop_side / 224)
        bottom = crop_y + int((y2 + 1) * crop_side / 224)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
    cv2.rectangle(frame, (8, max(0, height - 52)), (min(width - 8, 1100), height - 8), (0, 0, 0), -1)
    cv2.putText(frame, text[:130], (16, height - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)


def main(args):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Video inference requires OpenCV. Install it with `python -m pip install opencv-python`.") from exc

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1")
    if args.explain_every < 0:
        raise ValueError("--explain-every must be non-negative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    class_names = config.get("class_names", CLASSES)
    if len(class_names) != config["num_classes"]:
        raise ValueError("Checkpoint class_names length does not match num_classes")
    temperature, calibration = load_temperature(args.calibration, checkpoint_path, class_names)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {args.source}")

    output_path = Path(args.output_video).expanduser().resolve() if args.output_video else None
    explanation_path = Path(args.explanation_log).expanduser().resolve() if args.explain_every else None
    writer = None
    explanation_stream = None
    probability_history = deque(maxlen=args.smooth_window)
    started = time.perf_counter()
    frames = 0
    predicted_frames = 0
    inference_errors = 0
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if explanation_path:
            explanation_path.parent.mkdir(parents=True, exist_ok=True)
            explanation_stream = explanation_path.open("w", encoding="utf-8")

        print(f"Source: {args.source} | device: {device} | input: RGB 224x224 | press q or Esc to stop")
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                if output_path and writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {output_path}")
                display = frame.copy()
                elapsed = max(time.perf_counter() - started, 1e-6)
                throughput = frames / elapsed
                cv2.rectangle(display, (8, 8), (min(display.shape[1] - 8, 760), 76), (0, 0, 0), -1)
                try:
                    logits = model(frame_to_tensor(frame, device))
                    current_probs = calibrated_probabilities(logits, temperature)[0].cpu().numpy()
                    if not np.isfinite(current_probs).all():
                        raise RuntimeError("Model produced non-finite class probabilities")
                    probability_history.append(current_probs)
                    probabilities = np.mean(probability_history, axis=0)
                    order = np.argsort(probabilities)[::-1]
                    best, second = int(order[0]), int(order[1])
                    predicted_frames += 1
                    title = f"{class_names[best]}  {probabilities[best] * 100:.1f}%"
                    confidence_label = "temperature-scaled" if calibration else "softmax"
                    detail = (f"Next: {class_names[second]}  {probabilities[second] * 100:.1f}% | "
                              f"{confidence_label} score | {throughput:.1f} display FPS")
                    if args.explain_every and frames % args.explain_every == 0:
                        try:
                            target, score, heatmap, box, explanation = explain_frame(
                                model, frame, device, class_names, temperature)
                            draw_explanation_overlay(display, heatmap, box, explanation, cv2)
                            explanation_record = {
                                "frame": frames,
                                "predicted_class": class_names[target],
                                "displayed_probability": score,
                                "probability_type": confidence_label,
                                "candidate_box_xyxy_source_frame": list(box) if box is not None else None,
                                "text_explanation": explanation,
                                "localization_validated": False,
                            }
                        except Exception as explanation_error:
                            explanation_record = {
                                "frame": frames,
                                "explanation_status": "unavailable",
                                "error_type": type(explanation_error).__name__,
                                "prediction_suppressed": False,
                            }
                            if frames == args.explain_every:
                                print(f"Explanation generation failed ({type(explanation_error).__name__}); prediction continues.")
                        explanation_stream.write(json.dumps(explanation_record, allow_nan=False) + "\n")
                        explanation_stream.flush()
                    cv2.putText(display, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                (70, 255, 70), 2, cv2.LINE_AA)
                    cv2.putText(display, detail, (18, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (255, 255, 255), 1, cv2.LINE_AA)
                except Exception as exc:
                    inference_errors += 1
                    probability_history.clear()
                    cv2.putText(display, "AI UNAVAILABLE - prediction suppressed", (18, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, f"Frame shown without AI output | {throughput:.1f} display FPS",
                                (18, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                    if inference_errors == 1:
                        print(f"Frame inference failed ({type(exc).__name__}); continuing video without AI output.")
                if writer is not None:
                    writer.write(display)
                cv2.imshow("SpikeFusion-GI video classification", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if explanation_stream is not None:
            explanation_stream.close()
        cv2.destroyAllWindows()

    print(f"Displayed {frames} frames in {time.perf_counter() - started:.2f}s; "
          f"predictions={predicted_frames}, inference_errors={inference_errors}.")
    if output_path:
        print(f"Saved annotated video to {output_path}")
    if explanation_path:
        print(f"Saved frame explanations to {explanation_path}")
    if frames == 0:
        raise RuntimeError("The source opened but yielded no frames.")


def parse_args():
    parser = argparse.ArgumentParser(description="Classify video frames and display predictions")
    parser.add_argument("--source", default="0", help="Camera index (default 0), video path, or stream URL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default=None,
                        help="Optional validation-fitted calibration JSON for this exact checkpoint")
    parser.add_argument("--explain-every", type=int, default=0,
                        help="Generate Grad-CAM/SpikeCAM overlay and text every N frames; 0 disables the slower explanation pass")
    parser.add_argument("--explanation-log", default="video_explanations.jsonl",
                        help="JSONL path for explanations when --explain-every is enabled")
    parser.add_argument("--output-video", default=None, help="Optional path for an annotated MP4")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Average probabilities over this many recent frames; 1 disables smoothing")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
