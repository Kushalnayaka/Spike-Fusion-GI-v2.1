"""Model-agnostic image-region SHAP explanations for a trained checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from dataset import CLASSES
from model import build_model_from_config


def main(args):
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("SHAP is optional. Install it with `pip install shap>=0.48` and retry.") from exc

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image = ImageOps.fit(image, (224, 224), method=Image.Resampling.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    class_names = config.get("class_names", CLASSES)

    def predict_proba(images_hwc):
        array = np.asarray(images_hwc, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.inference_mode():
            return model(tensor).softmax(dim=1).cpu().numpy()

    probabilities = predict_proba(image_np)[0]
    target = int(probabilities.argmax()) if args.class_index is None else args.class_index
    if target < 0 or target >= len(class_names):
        raise ValueError(f"class-index must be in [0, {len(class_names) - 1}]")

    # Image masker groups pixels into a hierarchy and replaces hidden regions
    # with a blurred baseline; this is an approximate, model-agnostic SHAP run.
    masker = shap.maskers.Image(f"blur({args.blur},{args.blur})", shape=image_np.shape)
    explainer = shap.Explainer(predict_proba, masker, output_names=class_names,
                               algorithm="partition", seed=args.seed)
    values = explainer(image_np[None, ...], max_evals=args.max_evals, batch_size=args.batch_size)
    selected = values[..., target]

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.plots.image(selected, show=False)
    figure = plt.gcf()
    figure.savefig(out / "shap.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    np.savez_compressed(out / "shap_values.npz", values=np.asarray(selected.values),
                        base_values=np.asarray(selected.base_values),
                        image=image_np, class_index=np.asarray(target))
    result = {"method": "SHAP PartitionExplainer + Image masker",
              "mask_baseline": f"blur({args.blur},{args.blur})",
              "max_evals": args.max_evals, "class_index": target,
              "class_name": class_names[target], "probability": float(probabilities[target]),
              "note": "Approximate image-region attributions depend on the blur baseline and evaluation budget."}
    (out / "shap_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved SHAP plot and values to {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create approximate SHAP image-region attributions")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="xai_outputs/shap")
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--blur", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
