from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from calibration import calibrated_probabilities, load_temperature
from dataset import create_loaders
from metrics import compute_metrics
from model import build_model_from_config
from provenance import sha256_file


@torch.inference_mode()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else checkpoint_path.parent
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    saved_manifest_hash = checkpoint.get("provenance", {}).get("run_manifest_sha256")
    manifest_data = None
    training_result = None
    if saved_manifest_hash:
        if not manifest_path.is_file():
            raise FileNotFoundError("This checkpoint records run provenance, but run_manifest.json is missing")
        present_manifest_hash = sha256_file(manifest_path)
        if present_manifest_hash != saved_manifest_hash:
            raise ValueError("run_manifest.json hash does not match the checkpoint provenance")
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_provenance = checkpoint.get("provenance", {})
        if manifest_data.get("dataset", {}).get("sha256") != checkpoint_provenance.get("dataset_sha256"):
            raise ValueError("run_manifest.json dataset hash does not match checkpoint provenance")
        if manifest_data.get("code_sha256") != checkpoint_provenance.get("code_sha256"):
            raise ValueError("run_manifest.json source hash does not match checkpoint provenance")
        expected_split_hash = manifest_data.get("split_file", {}).get("sha256")
        split_path = run_dir / "splits.json"
        if not expected_split_hash:
            raise ValueError("Run manifest does not record the training split hash")
        if not split_path.is_file():
            raise FileNotFoundError(f"Training split file required for evaluation is missing: {split_path}")
        if sha256_file(split_path) != expected_split_hash:
            raise ValueError("splits.json hash does not match the training run manifest")
        result_path = checkpoint_path.parent / "training_result.json"
        if not result_path.is_file():
            raise FileNotFoundError("This checkpoint records run provenance, but training_result.json is missing")
        training_result = json.loads(result_path.read_text(encoding="utf-8"))
        present_checkpoint_hash = sha256_file(checkpoint_path)
        if training_result.get("checkpoint_sha256") != present_checkpoint_hash:
            raise ValueError("training_result.json checkpoint hash does not match the evaluated checkpoint")
        if training_result.get("checkpoint") != checkpoint_path.name:
            raise ValueError("training_result.json checkpoint name does not match the evaluated checkpoint")
        if training_result.get("run_manifest_sha256") != present_manifest_hash:
            raise ValueError("training_result.json manifest hash does not match run_manifest.json")
        if training_result.get("dataset_sha256") != checkpoint.get("provenance", {}).get("dataset_sha256"):
            raise ValueError("training_result.json dataset hash does not match checkpoint provenance")
        if training_result.get("best_epoch") != checkpoint.get("best_epoch"):
            raise ValueError("training_result.json selected epoch does not match the checkpoint")
    _, _, test_loader, class_names, splits, dataset_identity = create_loaders(
        args.data_root, run_dir, batch_size=args.batch_size, workers=args.workers, seed=config["seed"])
    if class_names != config.get("class_names", class_names):
        raise ValueError("Evaluation dataset class order differs from the trained checkpoint")
    temperature, calibration = load_temperature(args.calibration, checkpoint_path, class_names)
    training_dataset_hash = checkpoint.get("provenance", {}).get("dataset_sha256")
    if training_dataset_hash and training_dataset_hash != dataset_identity["sha256"]:
        raise ValueError("Evaluation dataset content differs from training; this script evaluates the saved internal test split")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    labels_all, probs_all, calibrated_probs_all, loss_sum, count = [], [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = logits.float().softmax(1)
        calibrated_probs = calibrated_probabilities(logits, temperature)
        loss_sum += float(loss) * labels.numel()
        count += labels.numel()
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        calibrated_probs_all.extend(calibrated_probs.cpu().tolist())
    metrics = compute_metrics(labels_all, probs_all, class_names)
    metrics["confidence_calibration"] = ("validation_temperature_scaled" if calibration
                                         else "raw_uncalibrated_softmax")
    metrics["calibration_file"] = str(Path(args.calibration).expanduser().resolve()) if calibration else None
    metrics["temperature"] = temperature if calibration else None
    if calibration:
        metrics["calibrated_metrics"] = compute_metrics(labels_all, calibrated_probs_all, class_names)
    metrics["loss"] = loss_sum / max(count, 1)
    if count != len(test_loader.dataset) or len(labels_all) != count or len(probs_all) != count:
        raise RuntimeError("Evaluation sample count differs from the saved test split")
    confusion = metrics["confusion_matrix"]
    if sum(sum(row) for row in confusion) != count:
        raise RuntimeError("Confusion matrix total differs from the evaluated test sample count")
    metrics["sample_count"] = count
    metrics["error_count"] = count - sum(confusion[index][index] for index in range(len(class_names)))
    metrics["misclassification_rate"] = metrics["error_count"] / count
    metrics["split"] = "test"
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["best_epoch"] = checkpoint["best_epoch"]
    present_manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    metrics["provenance"] = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "saved_run_manifest_sha256": saved_manifest_hash,
        "present_run_manifest_sha256": present_manifest_hash,
        "run_manifest_hash_matches_checkpoint": (saved_manifest_hash == present_manifest_hash
                                                   if saved_manifest_hash and present_manifest_hash else None),
        "training_code_sha256": checkpoint.get("provenance", {}).get("code_sha256"),
        "training_dataset_sha256": training_dataset_hash,
        "evaluated_dataset_sha256": dataset_identity["sha256"],
        "evaluation_dataset_matches_training": (training_dataset_hash == dataset_identity["sha256"]
                                                if training_dataset_hash else None),
        "test_split_count": len(splits["test"]),
        "split_file_sha256": sha256_file(run_dir / "splits.json"),
        "training_result": str(checkpoint_path.parent / "training_result.json") if training_result else None,
        "training_result_verified": training_result is not None,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_name(f"{output.stem}_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "relative_image_path", "true_class", "predicted_class", "correct",
            "true_class_probability", "predicted_class_probability", "runner_up_class",
            "runner_up_probability", "calibrated_predicted_class_probability",
            "class_probabilities_json",
        ])
        writer.writeheader()
        for relative_path, true_label, probability_row, calibrated_row in zip(
                test_loader.dataset.paths, labels_all, probs_all, calibrated_probs_all):
            probabilities = np.asarray(probability_row, dtype=np.float64)
            calibrated_probabilities_row = np.asarray(calibrated_row, dtype=np.float64)
            ranking = np.argsort(probabilities)[::-1]
            predicted_label, runner_up_label = int(ranking[0]), int(ranking[1])
            writer.writerow({
                "relative_image_path": relative_path,
                "true_class": class_names[true_label],
                "predicted_class": class_names[predicted_label],
                "correct": true_label == predicted_label,
                "true_class_probability": f"{probabilities[true_label]:.8f}",
                "predicted_class_probability": f"{probabilities[predicted_label]:.8f}",
                "runner_up_class": class_names[runner_up_label],
                "runner_up_probability": f"{probabilities[runner_up_label]:.8f}",
                "calibrated_predicted_class_probability": f"{calibrated_probabilities_row[predicted_label]:.8f}",
                "class_probabilities_json": json.dumps(
                    {name: float(probabilities[index]) for index, name in enumerate(class_names)},
                    separators=(",", ":"), allow_nan=False),
            })
    metrics["prediction_review_csv"] = str(predictions_path)
    metrics["prediction_review_csv_sha256"] = sha256_file(predictions_path)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved held-out test metrics to {output}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a selected checkpoint on the held-out test split")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", default=None,
                   help="Optional validation-fitted calibration JSON for this exact checkpoint")
    p.add_argument("--output", default="test_metrics.json")
    p.add_argument("--run-dir", default=None, help="Directory containing the saved splits.json (default: checkpoint folder)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
