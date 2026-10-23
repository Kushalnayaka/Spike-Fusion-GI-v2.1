"""Fit scalar temperature scaling on the saved validation split only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import create_loaders
from model import build_model_from_config
from provenance import sha256_file


@torch.no_grad()
def collect_validation_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        logits_all.append(logits.float())
        labels_all.append(labels.to(device))
    if not logits_all:
        raise ValueError("Validation split is empty; cannot fit confidence calibration")
    return torch.cat(logits_all), torch.cat(labels_all)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find one positive scalar temperature minimizing validation cross-entropy."""
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp().cpu())


def main(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else checkpoint_path.parent
    provenance = checkpoint.get("provenance", {})
    saved_manifest_hash = provenance.get("run_manifest_sha256")
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if saved_manifest_hash:
        if not manifest_path.is_file() or sha256_file(manifest_path) != saved_manifest_hash:
            raise ValueError("Checkpoint run manifest is missing or fails its saved hash")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_split_hash = manifest.get("split_file", {}).get("sha256")
        split_path = run_dir / "splits.json"
        if not split_path.is_file() or sha256_file(split_path) != expected_split_hash:
            raise ValueError("Calibration split does not match the checkpoint's saved validation split")
    _, val_loader, _, class_names, _, dataset_identity = create_loaders(
        args.data_root, run_dir, batch_size=args.batch_size, workers=args.workers,
        seed=config["seed"])
    if class_names != config.get("class_names", class_names):
        raise ValueError("Calibration dataset class order differs from checkpoint")
    training_hash = checkpoint.get("provenance", {}).get("dataset_sha256")
    if training_hash and training_hash != dataset_identity["sha256"]:
        raise ValueError("Calibration must use the checkpoint's original dataset and saved validation split")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits, labels = collect_validation_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format": "spikefusion-temperature-calibration-v1",
        "fit_split": "validation",
        "method": "single positive scalar temperature scaling",
        "temperature": temperature,
        "sample_count": int(labels.numel()),
        "class_names": class_names,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_sha256": dataset_identity["sha256"],
        "split_file_sha256": sha256_file(run_dir / "splits.json"),
        "interpretation": "Changes probability calibration only; does not change argmax class predictions or accuracy.",
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Fit confidence temperature on the saved validation split")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None,
                        help="Run directory containing the saved splits.json (default: checkpoint folder)")
    parser.add_argument("--output", default="confidence_calibration.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
