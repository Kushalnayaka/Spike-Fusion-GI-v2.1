from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from model import build_model, build_model_from_config, profile_config


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)


def main(args):
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        cfg = checkpoint["model_config"]
        if args.profile is not None and args.profile != cfg.get("profile", cfg["sequence_mode"]):
            raise ValueError("Requested profile differs from checkpoint architecture; train the requested profile first")
        profile = cfg.get("profile", cfg["sequence_mode"])
        model = build_model_from_config(cfg)
        model.load_state_dict(checkpoint["model_state"])
        model_config = cfg
    else:
        profile = args.profile or "edge"
        model = build_model(profile)
        total, trainable = model.count_parameters()
        model_config = profile_config(profile)
        model_config.update({"architecture": "SpikeFusion-GI-v2.1-native-selective-scan",
                             "input_shape": [3, 224, 224], "input_range": [0.0, 1.0],
                             "retinal_channels": 10,
                             "fusion_channels": model.snn_out_channels + model.cnn.out_channels,
                             "parameter_count_total": total, "parameter_count_trainable": trainable})
    if args.dynamic_int8:
        if model.sequence_mode != "edge":
            print("Warning: a selective scan is left in floating point; for portable quantized scan-free export train --profile edge.")
        model = torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    model.eval()
    wrapper = LogitsOnly(model).eval()
    sample = torch.zeros(1, 3, 224, 224)
    traced = torch.jit.trace(wrapper, sample, strict=False, check_trace=False)
    traced.save(str(out / "model_torchscript.pt"))
    with torch.inference_mode():
        logits = traced(sample)
        dynamic_batch_logits = traced(sample.repeat(2, 1, 1, 1))
    if tuple(dynamic_batch_logits.shape) != (2, 8):
        raise RuntimeError(f"TorchScript dynamic-batch check failed: {tuple(dynamic_batch_logits.shape)}")
    manifest = {"format": "torchscript_trace", "input": "float32 RGB [B,3,224,224] in [0,1]",
                "output": "logits [B,8]", "spatial_size": 224, "batch_dynamic": True,
                "profile": profile, "sequence_mode": model.sequence_mode, "dynamic_int8_linear": args.dynamic_int8,
                "model_config": model_config, "torch_version": torch.__version__,
                "smoke_logits_shape": list(logits.shape), "onnx": "not attempted"}
    if args.try_onnx:
        try:
            onnx_sample = torch.zeros(2, 3, 224, 224)
            torch.onnx.export(wrapper, (onnx_sample,), str(out / "model.onnx"), input_names=["image"],
                              output_names=["logits"], dynamo=True,
                              dynamic_shapes={"image": {0: torch.export.Dim("batch", min=2)}})
            manifest["onnx"] = "exported"
        except Exception as exc:
            manifest["onnx"] = f"failed: {type(exc).__name__}: {exc}"
            print(f"ONNX export unavailable; TorchScript artifact is ready. {manifest['onnx']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved deployment artifacts in {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Export SpikeFusion to TorchScript and optionally ONNX")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="export")
    p.add_argument("--profile", choices=("mamba", "edge", "micro"), default=None)
    p.add_argument("--dynamic-int8", action="store_true", help="Dynamically quantize Linear layers for CPU inference")
    p.add_argument("--try-onnx", action="store_true", help="Attempt ONNX export if onnx/onnxscript are installed")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
