"""Requested fast model/SpikeCAM smoke check; no dataset needed."""
from __future__ import annotations

import json

import torch

from model import build_model


def spikecam(score, activation):
    grad = torch.autograd.grad(score, activation, retain_graph=False)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    lo, hi = cam.min(), cam.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(cam)
    return (cam - lo) / (hi - lo)


def main():
    torch.manual_seed(7)
    report = {}
    for profile in ("mamba", "edge", "micro"):
        model = build_model(profile).eval()
        total, trainable = model.count_parameters()
        assert total <= 1_100_000, f"{profile} parameter count {total:,} exceeds 1.1M"
        if profile == "micro":
            assert total <= 30_000, f"micro profile is no longer compact: {total:,} parameters"
        image = torch.rand(1, 3, 224, 224)
        logits, aux = model.forward_with_aux(image)
        assert logits.shape == (1, 8), tuple(logits.shape)
        assert aux["retinal_map"].shape == (1, 10, 224, 224)
        assert aux["spike_train"].shape[1] == model.snn.steps
        cam = spikecam(logits[0, 0], aux["snn_map"])
        assert cam.shape == (1, 1, 224, 224)
        assert bool(torch.isfinite(cam).all())
        report[profile] = {"parameters_total": total, "parameters_trainable": trainable,
                           "logits_shape": list(logits.shape), "spikecam_shape": list(cam.shape),
                           "spikecam_range": [float(cam.min()), float(cam.max())]}
    print(json.dumps(report, indent=2))
    print("PASS: all model profiles forward, retinal output, parameter budgets, and gradient-based SpikeCAM")


if __name__ == "__main__":
    main()
