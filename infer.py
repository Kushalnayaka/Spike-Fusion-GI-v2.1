from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from calibration import calibrated_probabilities, load_temperature
from dataset import CLASSES
from model import build_model_from_config


def load_image(path: Path, size=224):
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return image, tensor


def make_cam(score: torch.Tensor, activation: torch.Tensor, retain_graph: bool = False) -> np.ndarray:
    grad = torch.autograd.grad(score, activation, retain_graph=retain_graph,
                               allow_unused=False)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear",
                                          align_corners=False)[0, 0]
    cam = cam.detach().float().cpu().numpy()
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(cam.min()), float(cam.max())
    if high - low < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - low) / (high - low)).astype(np.float32)


def map_is_usable(heatmap: np.ndarray) -> bool:
    return bool(np.isfinite(heatmap).all() and float(heatmap.max() - heatmap.min()) > 1e-8)


def top_region_mask(heatmap: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    positive = heatmap[heatmap > 0]
    if positive.size == 0:
        return np.zeros_like(heatmap, dtype=bool)
    cutoff = float(np.quantile(positive, 1.0 - fraction))
    return heatmap >= max(cutoff, 0.35)


def largest_component_box(mask: np.ndarray):
    """Return an (x1, y1, x2, y2) box around the largest 8-connected region."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(y, x)])
        component = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    if not best:
        return None
    ys = [point[0] for point in best]
    xs = [point[1] for point in best]
    return max(0, min(xs)), max(0, min(ys)), min(w - 1, max(xs)), min(h - 1, max(ys))


def save_overlay(image: Image.Image, heatmap: np.ndarray, path: Path, box=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heat = (plt.get_cmap("jet")(np.clip(heatmap, 0, 1))[..., :3] * 255).astype(np.uint8)
    color = Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR)
    result = Image.blend(image.convert("RGB"), color, 0.42)
    if box is not None:
        draw = ImageDraw.Draw(result)
        draw.rectangle(box, outline=(255, 40, 40), width=3)
    result.save(path)


def save_box_image(image: Image.Image, box, path: Path):
    result = image.copy().convert("RGB")
    if box is not None:
        draw = ImageDraw.Draw(result)
        draw.rectangle(box, outline=(255, 30, 30), width=3)
        label = "candidate evidence region"
        draw.rectangle((box[0], max(0, box[1] - 16), box[0] + 150, box[1]), fill=(255, 30, 30))
        draw.text((box[0] + 2, max(0, box[1] - 15)), label, fill="white")
    result.save(path)


def tensor_from_pil(image: Image.Image, device):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


@torch.inference_mode()
def probabilities_for(model, image_tensor, temperature: float = 1.0):
    return calibrated_probabilities(model(image_tensor), temperature)[0].cpu().numpy()


def blur_box(image: Image.Image, box):
    blurred = image.filter(ImageFilter.GaussianBlur(radius=12))
    result = image.copy()
    x1, y1, x2, y2 = box
    pil_box = (x1, y1, x2 + 1, y2 + 1)
    result.paste(blurred.crop(pil_box), pil_box)
    return result


def occlusion_check(model, image: Image.Image, device, target: int, box,
                    baseline_probability: float, samples: int = 12, temperature: float = 1.0):
    if box is None:
        return {"status": "not_run", "reason": "No usable evidence box was produced."}
    w, h = image.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if bw >= w or bh >= h or bw * bh < 16:
        return {"status": "not_run", "reason": "Evidence box was too large or too small for a fair occlusion comparison."}

    target_masked = blur_box(image, box)
    target_probability = float(probabilities_for(
        model, tensor_from_pil(target_masked, device), temperature)[target])
    target_drop = baseline_probability - target_probability

    rng = np.random.default_rng(2026)
    random_drops = []
    for _ in range(samples):
        rx = int(rng.integers(0, w - bw + 1))
        ry = int(rng.integers(0, h - bh + 1))
        random_box = (rx, ry, rx + bw - 1, ry + bh - 1)
        random_image = blur_box(image, random_box)
        random_probability = float(probabilities_for(
            model, tensor_from_pil(random_image, device), temperature)[target])
        random_drops.append(baseline_probability - random_probability)
    random_median = float(np.median(random_drops))
    return {"status": "ran", "method": "blur candidate box and compare with equal-size random boxes",
            "target_probability_after_occlusion": target_probability,
            "target_probability_drop": float(target_drop),
            "random_box_probability_drop_median": random_median,
            "evidence_drop_exceeds_random_median": bool(target_drop > random_median),
            "random_control_samples": samples,
            "interpretation": "Sensitivity check only; blur can create distribution shift and does not validate lesion localization."}


def main(args):
    if args.occlusion_samples < 1:
        raise ValueError("--occlusion-samples must be at least 1")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    class_names = config.get("class_names", CLASSES)
    if len(class_names) != config["num_classes"]:
        raise ValueError("Checkpoint class_names length does not match num_classes")
    temperature, calibration = load_temperature(args.calibration, checkpoint_path, class_names)

    original, image = load_image(Path(args.image))
    image = image.to(device).requires_grad_(True)
    logits, aux = model.forward_with_aux(image)
    probabilities = calibrated_probabilities(logits, temperature)[0].detach().cpu().numpy()
    predicted_index = int(probabilities.argmax())
    target = predicted_index if args.class_index is None else args.class_index
    if target < 0 or target >= config["num_classes"]:
        raise ValueError(f"class-index must be in [0, {config['num_classes'] - 1}]")

    gradcam = make_cam(logits[0, target], aux["cnn_map"], retain_graph=True)
    spikecam = make_cam(logits[0, target], aux["snn_map"])
    grad_valid, spike_valid = map_is_usable(gradcam), map_is_usable(spikecam)
    available_maps = [m for m, valid in ((gradcam, grad_valid), (spikecam, spike_valid)) if valid]
    combined = np.mean(available_maps, axis=0) if available_maps else np.zeros((224, 224), dtype=np.float32)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    grad_mask, spike_mask = top_region_mask(gradcam), top_region_mask(spikecam)
    map_agreement = (float(np.logical_and(grad_mask, spike_mask).sum() /
                           max(np.logical_or(grad_mask, spike_mask).sum(), 1))
                     if grad_valid and spike_valid else None)
    box = largest_component_box(top_region_mask(combined)) if available_maps else None

    order = np.argsort(probabilities)[::-1]
    runner_up = next((int(index) for index in order if int(index) != target), target)
    confidence = float(probabilities[target])
    margin = float(probabilities[target] - probabilities[runner_up])
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))

    # Horizontal flips are part of the training augmentation, so this is a modest
    # input-stability check rather than an independent correctness test.
    flip_probs = probabilities_for(model, image.detach().flip(-1), temperature)
    flip_class = int(np.argmax(flip_probs))
    flip_consistent = flip_class == predicted_index

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_overlay(original, gradcam, out / "gradcam.png")
    save_overlay(original, spikecam, out / "spikecam.png")
    save_overlay(original, combined, out / "combined_heatmap.png", box=box)
    save_box_image(original, box, out / "evidence_box.png")

    occlusion = occlusion_check(model, original, device, target, box, confidence,
                                samples=args.occlusion_samples, temperature=temperature)
    box_fraction = (((box[2] - box[0] + 1) * (box[3] - box[1] + 1)) / (224 * 224)
                    if box is not None else None)
    validation = {
        "probabilities_finite_and_sum_to_one": bool(np.isfinite(probabilities).all() and
                                                     abs(float(probabilities.sum()) - 1.0) < 1e-4),
        "gradcam_usable": grad_valid,
        "spikecam_usable": spike_valid,
        "gradcam_spikecam_top_region_iou": map_agreement,
        "horizontal_flip_prediction_same": bool(flip_consistent),
        "horizontal_flip_predicted_class": class_names[flip_class],
        "evidence_box_area_fraction": float(box_fraction) if box_fraction is not None else None,
        "occlusion_sensitivity": occlusion,
        "clinical_localization_validated": False,
        "clinical_localization_note": "No pixel-level lesion masks are used; the box is a model-evidence region, not a validated lesion boundary.",
    }
    result = {
        "predicted_class": class_names[predicted_index],
        "explained_class": class_names[target], "class_index": target,
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
        "top_alternative": {"class": class_names[runner_up], "probability": float(probabilities[runner_up])},
        "top_probability_margin": margin,
        "softmax_entropy_nats": entropy,
        "confidence_calibration": ("validation_temperature_scaled; requires independent calibration assessment"
                                   if calibration else "not_calibrated; softmax score is not a validated clinical confidence"),
        "temperature": temperature if calibration else None,
        "calibration_file": str(Path(args.calibration).expanduser().resolve()) if calibration else None,
        "evidence_box_xyxy_pixels_224": list(box) if box is not None else None,
        "explanation_validation": validation,
        "explanation_scope": "Attribution and perturbation checks describe model sensitivity; they do not establish causal or clinical reasons.",
    }
    (out / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    pct = lambda value: f"{100.0 * value:.1f}%"
    if box is None:
        box_text = "No stable evidence box could be formed from the available heatmaps."
    else:
        box_text = (f"The candidate evidence box covers x={box[0]}..{box[2]}, y={box[1]}..{box[3]} "
                    f"on the 224x224 resized image ({pct(box_fraction)} of its area). It is not a lesion outline.")
    if map_agreement is None:
        agreement_text = "Grad-CAM/SpikeCAM overlap could not be compared because at least one map was unusable."
    else:
        agreement_text = f"The top-attribution regions from Grad-CAM and SpikeCAM have IoU {map_agreement:.3f}."
    if occlusion.get("status") == "ran":
        occlusion_text = (f"Blurring the candidate box changed the selected-class probability from {pct(confidence)} "
                          f"to {pct(occlusion['target_probability_after_occlusion'])} "
                          f"(drop {pct(occlusion['target_probability_drop'])}); the median drop for "
                          f"{args.occlusion_samples} equal-size random boxes was "
                          f"{pct(occlusion['random_box_probability_drop_median'])}. "
                          f"Evidence-region drop exceeded that median: "
                          f"{occlusion['evidence_drop_exceeds_random_median']}.")
    else:
        occlusion_text = f"Occlusion comparison was not run: {occlusion.get('reason', 'unknown reason')}"
    explanation = (
        f"Prediction: {class_names[predicted_index]} ({pct(probabilities[predicted_index])} "
        f"{'temperature-scaled score' if calibration else 'uncalibrated softmax score'}).\n"
        f"Explained class: {class_names[target]} ({pct(confidence)} "
        f"{'validation-temperature-scaled' if calibration else 'uncalibrated softmax'} score).\n"
        f"Next class: {class_names[runner_up]} ({pct(probabilities[runner_up])}; margin {pct(margin)}).\n"
        f"{box_text}\n{agreement_text}\n{occlusion_text}\n"
        f"Horizontal-flip check: {'same' if flip_consistent else 'different'} predicted class "
        f"({class_names[flip_class]} after flip).\n"
        "Interpretation: these checks describe where this model responds and how its score changes under perturbation. "
        "They do not prove that the boxed region is a lesion or that the prediction is medically correct. "
        "Clinical validation requires held-out annotated data and pixel-level lesion masks for localization.\n"
    )
    (out / "explanation.txt").write_text(explanation, encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved explanation report and overlays to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Classify an image and save CAM overlays, a candidate evidence box, and validation report")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", default=None,
                   help="Optional validation-fitted calibration JSON for this exact checkpoint")
    p.add_argument("--output-dir", default="xai_outputs")
    p.add_argument("--class-index", type=int, default=None)
    p.add_argument("--occlusion-samples", type=int, default=12)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
