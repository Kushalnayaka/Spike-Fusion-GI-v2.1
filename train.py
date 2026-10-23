from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import create_loaders
from metrics import compute_metrics
from model import build_model, profile_config
from provenance import project_code_sha256, sha256_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def mix_training_batch(images, labels, mixup_alpha: float, cutmix_alpha: float, probability: float):
    """Apply optional batch-level MixUp or CutMix; returns soft-target components."""
    if probability <= 0 or np.random.random() >= probability:
        return images, labels, None, 1.0, "none"

    choices = []
    if mixup_alpha > 0:
        choices.append("mixup")
    if cutmix_alpha > 0:
        choices.append("cutmix")
    if not choices or images.shape[0] < 2:
        return images, labels, None, 1.0, "none"

    mode = random.choice(choices)
    alpha = mixup_alpha if mode == "mixup" else cutmix_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    target_b = labels[permutation]

    if mode == "mixup":
        mixed = images * lam + images[permutation] * (1.0 - lam)
    else:
        height, width = images.shape[-2:]
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_width, cut_height = int(width * cut_ratio), int(height * cut_ratio)
        center_x = int(torch.randint(width, (1,), device=images.device).item())
        center_y = int(torch.randint(height, (1,), device=images.device).item())
        x1, x2 = max(center_x - cut_width // 2, 0), min(center_x + cut_width // 2, width)
        y1, y2 = max(center_y - cut_height // 2, 0), min(center_y + cut_height // 2, height)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))

    return mixed, labels, target_b, lam, mode


def epoch_lr_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_factor: float = 0.02) -> float:
    """Linear warmup followed by cosine decay, expressed as an LR multiplier."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return min((epoch + 1) / warmup_epochs, 1.0)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + np.cos(np.pi * progress))


@torch.inference_mode()
def evaluate_loader(model, loader, device, class_names):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss) * labels.numel()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(logits.float().softmax(1).cpu().tolist())
    metrics = compute_metrics(all_labels, all_probs, class_names)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


class ImageNetTeacher(nn.Module):
    """Training-only pretrained torchvision teacher; never included in export."""
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        pretrained = mobilenet_v3_small(weights=weights)
        self.features = pretrained.features
        self.classifier = nn.Linear(576, num_classes)
        self.register_buffer("mean", torch.tensor(weights.transforms().mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(weights.transforms().std).view(1, 3, 1, 1))

    def forward(self, image):
        features = self.features((image - self.mean) / self.std)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(pooled)


def train_transfer_teacher(args, train_loader, val_loader, class_names, device, out):
    try:
        teacher = ImageNetTeacher(len(class_names)).to(device)
    except Exception as exc:
        raise RuntimeError("Could not load torchvision MobileNetV3-Small ImageNet weights. "
                           "Check internet access/cache or omit --transfer-learning.") from exc
    teacher_total = sum(p.numel() for p in teacher.parameters())
    print(f"Transfer teacher: MobileNetV3-Small ({teacher_total:,} parameters), training-only; "
          "it is not part of the deployed student or its parameter budget.")
    teacher_epochs = min(args.transfer_teacher_epochs, args.epochs)
    warmup_epochs = min(2, teacher_epochs)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.transfer_teacher_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, stale = -1.0, 0
    best_path = out / "transfer_teacher.pt"
    for epoch in range(1, teacher_epochs + 1):
        train_head_only = epoch <= warmup_epochs
        for parameter in teacher.features.parameters():
            parameter.requires_grad_(not train_head_only)
        teacher.train()
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = teacher(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += labels.numel()
            total_loss += float(loss.detach()) * labels.numel()
        val_metrics = evaluate_loader(teacher, val_loader, device, class_names)
        print(f"Teacher epoch {epoch:02d}/{teacher_epochs} train_loss={total_loss / max(seen, 1):.4f} "
              f"val_macro_f1={val_metrics['f1_macro']:.4f}")
        if val_metrics["f1_macro"] > best_score:
            best_score, stale = val_metrics["f1_macro"], 0
            torch.save({"model_state": teacher.state_dict(), "best_val_macro_f1": best_score,
                        "teacher": "torchvision.mobilenet_v3_small.imagenet"}, best_path)
        else:
            stale += 1
            if stale >= args.transfer_teacher_patience:
                break
    teacher.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, str(best_path)


def train(args):
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.mixup_alpha < 0 or args.cutmix_alpha < 0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be non-negative")
    if not 0.0 <= args.mix_probability <= 1.0:
        raise ValueError("--mix-probability must be between 0 and 1")
    set_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(
            f"Run output path must be a new or empty directory to preserve prior experiment artifacts: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, _, class_names, splits, dataset_identity = create_loaders(
        args.data_root, out, batch_size=args.batch_size, workers=args.workers, seed=args.seed)
    config = profile_config(args.profile, num_classes=len(class_names))
    for name in ("token_dim", "ssm_depth", "state_dim", "snn_steps"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    model = build_model(args.profile, num_classes=len(class_names), **{
        key: value for key, value in config.items() if key not in {"profile", "num_classes"}
    }).to(device)
    total, trainable = model.count_parameters()
    if total > 1_100_000:
        raise RuntimeError(f"Parameter budget exceeded: {total:,} > 1,100,000")
    config.update({"architecture": "SpikeFusion-GI-v2.1-native-selective-scan",
                   "input_shape": [3, 224, 224], "input_range": [0.0, 1.0],
                   "retinal_channels": 10, "fusion_channels": config["snn_out_channels"] + config["cnn_widths"][-1],
                   "class_names": class_names, "seed": args.seed,
                   "parameter_count_total": total, "parameter_count_trainable": trainable,
                   "split_counts": {k: len(v) for k, v in splits.items()},
                   "transfer_learning": bool(args.transfer_learning),
                   "distillation_alpha": args.distillation_alpha if args.transfer_learning else None,
                   "distillation_temperature": args.distillation_temperature if args.transfer_learning else None})
    (out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device} | parameters: {total:,} total / {trainable:,} trainable")
    print("Split sizes: " + ", ".join(f"{name}={len(items)}" for name, items in splits.items()))

    project_root = Path(__file__).resolve().parent
    split_path = out / "splits.json"
    training_args = vars(args).copy()
    cuda_properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    run_manifest = {
        "format": "spikefusion-run-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": sys.argv,
        "training_args": training_args,
        "model_config": config,
        "dataset": {"root": str(Path(args.data_root).expanduser().resolve()), **dataset_identity},
        "split_file": {"name": split_path.name, "sha256": sha256_file(split_path)},
        "code_sha256": project_code_sha256(project_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "scikit_learn": package_version("scikit-learn"),
            "pillow": package_version("Pillow"),
            "cpu_logical_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": cuda_properties.name if cuda_properties is not None else None,
            "cuda_device_memory_bytes": int(cuda_properties.total_memory) if cuda_properties is not None else None,
            "cuda_compute_capability": ([cuda_properties.major, cuda_properties.minor]
                                        if cuda_properties is not None else None),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": True,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "mixed_precision_enabled": device.type == "cuda",
        },
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, allow_nan=False), encoding="utf-8")
    run_manifest_sha256 = sha256_file(manifest_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(max(args.warmup_epochs, 0), max(args.epochs - 1, 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: epoch_lr_factor(epoch, max(args.epochs, 1), warmup_epochs),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    teacher = None
    if args.transfer_learning:
        teacher, teacher_path = train_transfer_teacher(args, train_loader, val_loader, class_names, device, out)
        config["transfer_teacher_checkpoint"] = teacher_path
        (out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        run_manifest["model_config"] = config
        run_manifest["transfer_teacher_checkpoint"] = {
            "name": Path(teacher_path).name, "sha256": sha256_file(teacher_path)}
        manifest_path.write_text(json.dumps(run_manifest, indent=2, allow_nan=False), encoding="utf-8")
        run_manifest_sha256 = sha256_file(manifest_path)
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history_path = out / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            seen, loss_sum = 0, 0.0
            mixed_batches = {"mixup": 0, "cutmix": 0, "none": 0}
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                training_images, target_a, target_b, mix_lambda, mix_method = mix_training_batch(
                    images, labels, args.mixup_alpha, args.cutmix_alpha, args.mix_probability
                )
                mixed_batches[mix_method] += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(training_images)
                    supervised_loss = criterion(logits, target_a)
                    if target_b is not None:
                        supervised_loss = (mix_lambda * supervised_loss
                                           + (1.0 - mix_lambda) * criterion(logits, target_b))
                    if teacher is None:
                        loss = supervised_loss
                    else:
                        with torch.no_grad():
                            teacher_logits = teacher(training_images)
                        temperature = args.distillation_temperature
                        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                        student_log_probs = F.log_softmax(logits / temperature, dim=1)
                        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
                        loss = (1.0 - args.distillation_alpha) * supervised_loss + args.distillation_alpha * kd_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                seen += labels.numel()
                loss_sum += float(loss.detach()) * labels.numel()
            scheduler.step()
            val_metrics = evaluate_loader(model, val_loader, device, class_names)
            train_loss = loss_sum / max(seen, 1)
            row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics,
                   "mixed_batches": mixed_batches,
                   "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            score = val_metrics["f1_macro"]
            print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                  f"val_macro_f1={score:.4f} lr={optimizer.param_groups[0]['lr']:.2g}")
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                checkpoint = {"model_state": model.state_dict(), "model_config": config,
                              "best_epoch": best_epoch, "best_val_macro_f1": best_score,
                              "optimizer_state": optimizer.state_dict(),
                              "training_args": training_args,
                              "provenance": {"run_manifest_sha256": run_manifest_sha256,
                                             "dataset_sha256": dataset_identity["sha256"],
                                             "code_sha256": run_manifest["code_sha256"]}}
                torch.save(checkpoint, out / "best_model.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val macro-F1={best_score:.4f})")
                    break
    checkpoint_path = out / "best_model.pt"
    training_result = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_manifest_sha256": run_manifest_sha256,
        "dataset_sha256": dataset_identity["sha256"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "training_result.json").write_text(json.dumps(training_result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Best checkpoint: {out / 'best_model.pt'} (epoch {best_epoch}, val macro-F1={best_score:.4f})")
    print("Held-out test data was not used during training or checkpoint selection. Run evaluate.py separately.")


def parse_args():
    p = argparse.ArgumentParser(description="Train SpikeFusion-GI on Kvasir v2")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="runs/kvasir")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear learning-rate warmup before cosine decay; set 0 to disable")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="MixUp beta-distribution alpha; set 0 to disable MixUp")
    p.add_argument("--cutmix-alpha", type=float, default=1.0,
                   help="CutMix beta-distribution alpha; set 0 to disable CutMix")
    p.add_argument("--mix-probability", type=float, default=0.5,
                   help="Probability of applying MixUp or CutMix to a training batch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-dim", type=int, default=None)
    p.add_argument("--ssm-depth", type=int, default=None)
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--snn-steps", type=int, default=None)
    p.add_argument("--profile", choices=("mamba", "edge", "micro"), default="mamba")
    p.add_argument("--transfer-learning", action="store_true", help="Fine-tune ImageNet MobileNetV3-Small teacher, then distill into the compact student")
    p.add_argument("--transfer-teacher-epochs", type=int, default=10)
    p.add_argument("--transfer-teacher-patience", type=int, default=3)
    p.add_argument("--transfer-teacher-lr", type=float, default=1e-4)
    p.add_argument("--distillation-alpha", type=float, default=0.35)
    p.add_argument("--distillation-temperature", type=float, default=3.0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
