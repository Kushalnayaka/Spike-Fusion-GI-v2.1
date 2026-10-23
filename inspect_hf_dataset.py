"""Inspect a Hugging Face image dataset schema before adapting the trainer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


GROUP_TOKENS = ("patient", "procedure", "video", "case", "subject", "study", "source")
IMAGE_TOKENS = ("image", "frame", "pixel")
LABEL_TOKENS = ("label", "class", "category", "target", "diagnosis")


def _feature_type(feature: Any) -> str:
    return type(feature).__name__


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _profile_split(name: str, dataset: Any) -> dict[str, Any]:
    features = dataset.features
    columns = list(dataset.column_names)
    feature_types = {column: _feature_type(features[column]) for column in columns}
    image_columns = [column for column in columns
                     if feature_types[column].lower() == "image"
                     or any(token in column.lower() for token in IMAGE_TOKENS)]
    label_columns = [column for column in columns
                     if any(token in column.lower() for token in LABEL_TOKENS)]
    group_columns = [column for column in columns
                     if any(token in column.lower() for token in GROUP_TOKENS)]

    label_counts = {}
    for column in label_columns:
        feature = features[column]
        counts = Counter()
        values = dataset[column]
        label_feature = feature
        while hasattr(label_feature, "feature"):
            label_feature = label_feature.feature
        names = getattr(label_feature, "names", None)
        for value in values:
            label_values = value if isinstance(value, (list, tuple)) else [value]
            for label in label_values:
                if label is None:
                    key = "<null>"
                elif names is not None and isinstance(label, int) and 0 <= label < len(names):
                    key = str(names[label])
                else:
                    key = str(_jsonable(label))
                counts[key] += 1
        label_counts[column] = dict(sorted(counts.items()))

    return {
        "split": name,
        "rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "features": feature_types,
        "candidate_image_columns": image_columns,
        "candidate_label_columns": label_columns,
        "label_counts": label_counts,
        "label_count_semantics": "Label occurrences are counted; multi-label rows contribute once per listed label.",
        "candidate_group_id_columns": group_columns,
    }


def inspect_dataset(args) -> dict[str, Any]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise SystemExit("Hugging Face inspection requires `python -m pip install datasets`.") from exc

    load_kwargs = {"path": args.dataset_id}
    if args.config:
        load_kwargs["name"] = args.config
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.split:
        load_kwargs["split"] = args.split

    loaded = load_dataset(**load_kwargs)
    if isinstance(loaded, DatasetDict):
        split_reports = [_profile_split(name, split) for name, split in loaded.items()]
    else:
        split_reports = [_profile_split(args.split or "loaded", loaded)]
    return {
        "format": "spikefusion-huggingface-preflight-v1",
        "dataset_id": args.dataset_id,
        "config": args.config,
        "requested_revision": args.revision,
        "trust_remote_code_enabled": bool(args.trust_remote_code),
        "splits": split_reports,
        "limitations": [
            "This is schema and label-count discovery only; it does not validate medical labels or data rights.",
            "Loading the dataset may download data shards into the Hugging Face cache and use network bandwidth/storage.",
            "Candidate group columns are identified by name only; their semantics and stability require confirmation.",
            "A dataset fingerprint is not a substitute for a pinned dataset revision or content-level image hashes.",
            "This tool does not modify, export, or train on dataset images.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Hugging Face image/label fields, class counts, and candidate grouping columns"
    )
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset repository ID")
    parser.add_argument("--config", default=None, help="Optional dataset configuration name")
    parser.add_argument("--revision", default=None,
                        help="Pinned branch, tag, or commit; use a commit SHA for reproducible inspection")
    parser.add_argument("--split", default=None, help="Inspect one split; by default inspect available splits")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow dataset-provided Python code; only use for a reviewed, trusted repository")
    parser.add_argument("--output", default=None, help="Optional JSON report path; defaults to stdout")
    args = parser.parse_args()
    report = inspect_dataset(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved dataset preflight report: {output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
