"""Evaluate a provenance-enabled checkpoint on a separately sourced image set."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASSES, KvasirDataset, fingerprint_dataset, validate_data_root
from metrics import compute_metrics
from model import build_model_from_config
from provenance import project_code_sha256, sha256_file


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@torch.inference_mode()
def main(args):
    if not args.confirm_independent:
        raise ValueError("Pass --confirm-independent only after confirming this source was held out from model development")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    class_names = config.get("class_names")
    if class_names != CLASSES or config.get("num_classes") != len(CLASSES):
        raise ValueError("Checkpoint class names/order must exactly match the supported Kvasir class folders")

    run_dir = checkpoint_path.parent
    manifest_path = run_dir / "run_manifest.json"
    split_path = run_dir / "splits.json"
    result_path = run_dir / "training_result.json"
    checkpoint_provenance = checkpoint.get("provenance", {})
    expected_manifest_hash = checkpoint_provenance.get("run_manifest_sha256")
    if not expected_manifest_hash:
        raise ValueError("External evaluation requires a provenance-enabled training checkpoint")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError("Training run manifest is missing or its hash does not match the checkpoint")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not split_path.is_file() or sha256_file(split_path) != manifest.get("split_file", {}).get("sha256"):
        raise ValueError("Training split file is missing or its hash does not match the run manifest")
    if manifest.get("dataset", {}).get("sha256") != checkpoint_provenance.get("dataset_sha256"):
        raise ValueError("Training run manifest dataset hash does not match checkpoint provenance")
    if manifest.get("code_sha256") != checkpoint_provenance.get("code_sha256"):
        raise ValueError("Training run manifest source hash does not match checkpoint provenance")
    if not result_path.is_file():
        raise FileNotFoundError("Training result record is missing; external evaluation requires a completed training run")
    training_result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint_path)
    if training_result.get("checkpoint") != checkpoint_path.name:
        raise ValueError("Training result checkpoint name does not match the supplied checkpoint")
    if training_result.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Training result checkpoint hash does not match the supplied checkpoint")
    if training_result.get("run_manifest_sha256") != expected_manifest_hash:
        raise ValueError("Training result manifest hash does not match the checkpoint provenance")
    if training_result.get("dataset_sha256") != checkpoint_provenance.get("dataset_sha256"):
        raise ValueError("Training result dataset hash does not match checkpoint provenance")
    if training_result.get("best_epoch") != checkpoint.get("best_epoch"):
        raise ValueError("Training result selected epoch does not match the supplied checkpoint")

    external_root, found_classes, by_class = validate_data_root(args.data_root)
    if found_classes != class_names:
        raise ValueError("External dataset class order differs from the trained checkpoint")
    external_identity = fingerprint_dataset(external_root, by_class)
    training_dataset_hash = checkpoint_provenance.get("dataset_sha256")
    if external_identity["sha256"] == training_dataset_hash:
        raise ValueError("External dataset fingerprint is identical to the training dataset")
    training_file_records = manifest.get("dataset", {}).get("files")
    if not isinstance(training_file_records, list) or len(training_file_records) != manifest.get("dataset", {}).get("image_count"):
        raise ValueError("Training manifest lacks the per-image hashes required for an exact-overlap audit")
    training_image_hashes = {record.get("sha256") for record in training_file_records}
    external_image_hashes = {record["sha256"] for record in external_identity["files"]}
    exact_overlap = training_image_hashes & external_image_hashes
    if exact_overlap:
        raise ValueError(
            f"External dataset contains {len(exact_overlap)} exact image-byte match(es) from the training cohort; "
            "it cannot be treated as fully independent."
        )

    relative_paths = [
        path.relative_to(external_root).as_posix()
        for class_name in class_names
        for path in by_class[class_name]
    ]
    dataset = KvasirDataset(external_root, relative_paths, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels_all, probabilities_all = [], []
    criterion = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    loss_sum, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = logits.float().softmax(dim=1)
        if not torch.isfinite(probs).all():
            raise RuntimeError("Model produced non-finite probabilities during external evaluation")
        loss_sum += float(loss) * labels.numel()
        count += labels.numel()
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(probs.cpu().tolist())

    if count == 0:
        raise RuntimeError("External dataset contains no evaluable images")
    metrics = compute_metrics(labels_all, probabilities_all, class_names)
    if sum(sum(row) for row in metrics["confusion_matrix"]) != count:
        raise RuntimeError("Confusion matrix total does not match the number of evaluated external images")
    if sum(values["support"] for values in metrics["per_class"].values()) != count:
        raise RuntimeError("Per-class support total does not match the number of evaluated external images")
    metrics["class_names"] = class_names
    metrics["loss"] = loss_sum / count
    metrics["sample_count"] = count
    metrics["evaluation_type"] = "external_full_dataset"
    metrics["dataset_id"] = args.dataset_id
    metrics["provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "independence_attestation": "asserted by invoker; not independently verified",
        "external_dataset_root": str(external_root),
        "external_dataset_sha256": external_identity["sha256"],
        "external_dataset_fingerprint_algorithm": external_identity["algorithm"],
        "external_dataset_image_count": external_identity["image_count"],
        "external_class_counts": external_identity["class_counts"],
        "exact_image_overlap_check": "passed; zero shared file-byte hashes",
        "exact_image_overlap_count": 0,
        "training_dataset_sha256": training_dataset_hash,
        "checkpoint_sha256": checkpoint_hash,
        "training_run_manifest_sha256": expected_manifest_hash,
        "training_split_sha256": sha256_file(split_path),
        "training_code_sha256": checkpoint_provenance.get("code_sha256"),
        "external_evaluator_code_sha256": project_code_sha256(Path(__file__).resolve().parent),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "numpy": package_version("numpy"),
        "scikit_learn": package_version("scikit-learn"),
        "pillow": package_version("Pillow"),
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mixed_precision_enabled": amp,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved external evaluation report to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on an independent full image dataset")
    parser.add_argument("--data-root", required=True, help="External dataset root with the same eight class folders")
    parser.add_argument("--dataset-id", required=True, help="Source and version identifier for the external cohort")
    parser.add_argument("--confirm-independent", action="store_true",
                        help="Attest that this dataset was held out from training, tuning, and model selection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="external_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
