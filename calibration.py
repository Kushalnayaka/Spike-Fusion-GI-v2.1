"""Validation-fitted temperature scaling for confidence reporting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from provenance import sha256_file


def load_temperature(calibration_path: str | Path | None, checkpoint_path: str | Path,
                    class_names: list[str]) -> tuple[float, dict | None]:
    """Load a calibrator only when it was fitted for this exact checkpoint/classes."""
    if calibration_path is None:
        return 1.0, None
    calibration_path = Path(calibration_path).expanduser().resolve()
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if data.get("format") != "spikefusion-temperature-calibration-v1":
        raise ValueError("Unsupported or malformed calibration file")
    expected_checkpoint_hash = sha256_file(Path(checkpoint_path).expanduser().resolve())
    if data.get("checkpoint_sha256") != expected_checkpoint_hash:
        raise ValueError("Calibration file was fitted for a different checkpoint")
    if data.get("class_names") != list(class_names):
        raise ValueError("Calibration file class order differs from the checkpoint")
    if data.get("fit_split") != "validation":
        raise ValueError("Only a validation-fitted calibration file can be used")
    temperature = float(data.get("temperature", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    return temperature, data


def calibrated_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return torch.softmax(logits.float() / float(temperature), dim=-1)
