"""Kvasir image-folder data loading and persisted near-duplicate-aware splits."""
from __future__ import annotations

import hashlib
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

CLASSES = ["dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", "normal-cecum",
           "normal-pylorus", "normal-z-line", "polyps", "ulcerative-colitis"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _dhash(path: Path) -> int:
    with Image.open(path) as im:
        gray = ImageOps.grayscale(im).resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


class _BKNode:
    def __init__(self, value: int):
        self.value = value
        self.children: Dict[int, "_BKNode"] = {}


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _near_duplicate_groups(paths: List[Path], radius: int = 3) -> List[List[Path]]:
    """Cluster perceptual hashes across classes; clusters stay intact across splits."""
    if not paths:
        return []
    hashes = [_dhash(p) for p in paths]
    parents = list(range(len(paths)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    root = _BKNode(hashes[0])
    seen: Dict[int, List[int]] = {hashes[0]: [0]}
    for idx, value in enumerate(hashes[1:], start=1):
        if value in seen:
            for other in seen[value]:
                union(idx, other)
            seen[value].append(idx)
            continue
        stack = [root]
        while stack:
            node = stack.pop()
            dist = _hamming(value, node.value)
            if dist <= radius:
                for other in seen[node.value]:
                    union(idx, other)
            low, high = dist - radius, dist + radius
            stack.extend(child for edge, child in node.children.items() if low <= edge <= high)
        node = root
        while True:
            dist = _hamming(value, node.value)
            if dist in node.children:
                node = node.children[dist]
            else:
                node.children[dist] = _BKNode(value)
                break
        seen.setdefault(value, []).append(idx)
    groups: Dict[int, List[Path]] = {}
    for i, path in enumerate(paths):
        groups.setdefault(find(i), []).append(path)
    return list(groups.values())


def validate_data_root(root: str | Path) -> Tuple[Path, List[str], Dict[str, List[Path]]]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    found = sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = sorted(set(CLASSES) - set(found))
    extra = sorted(set(found) - set(CLASSES))
    if missing or extra:
        raise ValueError(f"Expected exactly the 8 Kvasir class folders. Missing={missing}; extra={extra}")
    by_class = {}
    for cls in CLASSES:
        paths = sorted(p for p in (root / cls).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if len(paths) < 3:
            raise ValueError(f"Class '{cls}' has {len(paths)} readable image candidates; expected at least 3")
        bad = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                bad.append(path)
        if bad:
            raise ValueError(f"Unreadable image(s) in {cls}: {bad[:5]}")
        by_class[cls] = paths
    return root, CLASSES.copy(), by_class


def fingerprint_dataset(root: Path, by_class: Dict[str, List[Path]]) -> Dict:
    """Hash dataset identity and retain per-image digests for exact-overlap audits."""
    digest = hashlib.sha256()
    counts = {}
    files = []
    for class_name in CLASSES:
        paths = sorted(by_class[class_name], key=lambda path: path.relative_to(root).as_posix())
        counts[class_name] = len(paths)
        for path in paths:
            relative_path = path.relative_to(root).as_posix()
            relative = relative_path.encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            image_digest = hashlib.sha256()
            size_bytes = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    image_digest.update(block)
                    size_bytes += len(block)
            digest.update(b"\0")
            files.append({"class_name": class_name, "relative_path": relative_path,
                          "sha256": image_digest.hexdigest(), "size_bytes": size_bytes})
    return {"algorithm": "sha256(sorted relative path + image bytes)",
            "sha256": digest.hexdigest(), "image_count": sum(counts.values()),
            "class_counts": counts, "files": files}


def load_group_metadata(root: Path, by_class: Dict[str, List[Path]]):
    """Read optional pseudonymous patient/procedure/video groups from groups.csv."""
    metadata_path = root / "groups.csv"
    if not metadata_path.is_file():
        return None, None

    expected = {path.relative_to(root).as_posix()
                for paths in by_class.values() for path in paths}
    group_ids = {}
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"relative_path", "group_id"}.issubset(reader.fieldnames):
            raise ValueError("groups.csv must have relative_path and group_id columns")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("relative_path") or "").strip().replace("\\", "/")
            relative_path = Path(raw_path)
            normalized = Path(*raw_path.split("/")).as_posix()
            if (not raw_path or relative_path.is_absolute()
                    or any(part in {"", ".", ".."} for part in raw_path.split("/"))):
                raise ValueError(f"groups.csv row {row_number} has an unsafe or empty relative_path")
            if normalized not in expected:
                raise ValueError(f"groups.csv row {row_number} references an unknown image: {raw_path}")
            if normalized in group_ids:
                raise ValueError(f"groups.csv contains duplicate relative_path: {normalized}")
            group_id = (row.get("group_id") or "").strip()
            if not group_id:
                raise ValueError(f"groups.csv row {row_number} has an empty group_id")
            group_ids[normalized] = group_id

    missing = expected - set(group_ids)
    if missing:
        raise ValueError(f"groups.csv must include every image; missing {len(missing)} row(s), e.g. {sorted(missing)[:3]}")
    return group_ids, hashlib.sha256(metadata_path.read_bytes()).hexdigest()


def _combine_duplicate_and_metadata_groups(root: Path, paths: List[Path], group_ids=None):
    if not group_ids:
        return _near_duplicate_groups(paths)

    index = {path.relative_to(root).as_posix(): i for i, path in enumerate(paths)}
    parents = list(range(len(paths)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    first_by_group = {}
    for relative_path, group_id in group_ids.items():
        current = index[relative_path]
        if group_id in first_by_group:
            union(current, first_by_group[group_id])
        else:
            first_by_group[group_id] = current
    for duplicate_group in _near_duplicate_groups(paths):
        first = index[duplicate_group[0].relative_to(root).as_posix()]
        for path in duplicate_group[1:]:
            union(first, index[path.relative_to(root).as_posix()])

    groups = {}
    for i, path in enumerate(paths):
        groups.setdefault(find(i), []).append(path)
    return list(groups.values())


def make_or_load_splits(root: Path, by_class: Dict[str, List[Path]], split_file: Path,
                        seed: int = 42, ratios=(0.70, 0.15, 0.15),
                        group_ids=None, group_metadata_sha256=None) -> Dict[str, List[str]]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must be three positive values summing to 1; received {ratios}")
    relpaths = {p.relative_to(root).as_posix() for paths in by_class.values() for p in paths}
    if split_file.exists():
        data = json.loads(split_file.read_text(encoding="utf-8"))
        saved_ratios = data.get("ratios")
        if (data.get("seed") != seed or not isinstance(saved_ratios, list)
                or len(saved_ratios) != len(ratios)
                or any(abs(saved - requested) > 1e-9 for saved, requested in zip(saved_ratios, ratios))):
            raise ValueError(f"Existing {split_file} uses a different seed/ratio; choose another output directory")
        splits = data["splits"]
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("Persisted splits must contain exactly train, val, and test")
        combined = [p for name in ("train", "val", "test") for p in splits[name]]
        if len(combined) != len(set(combined)):
            raise ValueError("Persisted splits contain a path in multiple sets")
        if set(combined) != relpaths:
            raise ValueError("Persisted splits do not match this dataset; archive/remove splits.json or use a new output directory")
        if data.get("counts") != {name: len(splits[name]) for name in ("train", "val", "test")}:
            raise ValueError("Persisted split counts do not match the split assignments")
        if "class_counts" not in data:
            raise ValueError(
                f"Existing {split_file} predates per-class split validation; "
                "use a fresh output directory to create a newly recorded split."
            )
        if data.get("class_counts") != _split_class_counts(splits):
            raise ValueError("Persisted per-class split counts do not match the split assignments")
        if data.get("group_metadata_sha256") != group_metadata_sha256:
            raise ValueError("Persisted splits were created with different group metadata; use a fresh output directory")
        _require_class_coverage(splits)
        return splits

    rng = random.Random(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    all_paths = [path for cls in CLASSES for path in by_class[cls]]
    groups = _combine_duplicate_and_metadata_groups(root, all_paths, group_ids)
    rng.shuffle(groups)
    # Largest clusters first; keep each duplicate cluster whole while balancing
    # each class independently toward the requested proportions.
    groups.sort(key=len, reverse=True)
    targets = {cls: [len(by_class[cls]) * ratio for ratio in ratios] for cls in CLASSES}
    assigned = {cls: [0, 0, 0] for cls in CLASSES}
    for group in groups:
        class_counts = {cls: sum(1 for path in group if path.relative_to(root).parts[0] == cls)
                        for cls in CLASSES}
        slot = min(range(3), key=lambda i: sum(
            assigned[cls][i] / max(targets[cls][i], 1.0)
            for cls, count in class_counts.items() if count))
        split_name = ("train", "val", "test")[slot]
        splits[split_name].extend(path.relative_to(root).as_posix() for path in group)
        for cls, count in class_counts.items():
            assigned[cls][slot] += count
    for name in splits:
        splits[name].sort()
    _require_class_coverage(splits)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "ratios": list(ratios), "train_ratio": ratios[0],
               "near_duplicate_method": "global 64-bit dHash, Hamming radius <= 3",
               "group_metadata_file": "groups.csv" if group_ids is not None else None,
               "group_metadata_sha256": group_metadata_sha256,
               "grouping_policy": ("connected components of patient/procedure/video group_id and near-duplicate links"
                                    if group_ids is not None else "near-duplicate links only; no patient/procedure separation"),
               "counts": {k: len(v) for k, v in splits.items()},
               "class_counts": _split_class_counts(splits), "splits": splits}
    split_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits


def _split_class_counts(splits: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    return {
        split_name: {
            class_name: sum(Path(path).parts[0] == class_name for path in paths)
            for class_name in CLASSES
        }
        for split_name, paths in splits.items()
    }


def _require_class_coverage(splits: Dict[str, List[str]]) -> None:
    counts = _split_class_counts(splits)
    missing = {
        split_name: [class_name for class_name, count in class_counts.items() if count == 0]
        for split_name, class_counts in counts.items()
    }
    missing = {name: classes for name, classes in missing.items() if classes}
    if missing:
        raise ValueError(
            "Each train/val/test split must contain at least one image from every class; "
            f"missing class support: {missing}. Use a sufficiently large dataset or revise "
            "the duplicate-grouping/split protocol before training."
        )


class KvasirDataset(Dataset):
    def __init__(self, root: Path, paths: List[str], augment: bool = False, size: int = 224):
        self.root = root
        self.paths = paths
        self.augment = augment
        self.size = size
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        relpath = self.paths[index]
        image = Image.open(self.root / Path(relpath)).convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
            if random.random() < 0.6:
                image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
            for enhancer in (ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Brightness):
                if random.random() < 0.5:
                    image = enhancer(image).enhance(random.uniform(0.85, 1.15))
        image = ImageOps.fit(image, (self.size, self.size), method=Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        label = self.class_to_idx[Path(relpath).parts[0]]
        return tensor, label


def create_loaders(root: str | Path, output_dir: str | Path, batch_size: int = 32,
                   workers: int = 4, seed: int = 42):
    root, class_names, by_class = validate_data_root(root)
    dataset_identity = fingerprint_dataset(root, by_class)
    group_ids, group_metadata_sha256 = load_group_metadata(root, by_class)
    dataset_identity["group_metadata_sha256"] = group_metadata_sha256
    splits = make_or_load_splits(root, by_class, Path(output_dir) / "splits.json", seed=seed,
                                 group_ids=group_ids,
                                 group_metadata_sha256=group_metadata_sha256)
    train_ds = KvasirDataset(root, splits["train"], augment=True)
    val_ds = KvasirDataset(root, splits["val"])
    test_ds = KvasirDataset(root, splits["test"])
    generator = torch.Generator().manual_seed(seed)
    opts = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **opts)
    val_loader = DataLoader(val_ds, shuffle=False, **opts)
    test_loader = DataLoader(test_ds, shuffle=False, **opts)
    return train_loader, val_loader, test_loader, class_names, splits, dataset_identity
