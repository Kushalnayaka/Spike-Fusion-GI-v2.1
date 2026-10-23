# SpikeFusion-GI Baseline Protocol

## Purpose

Establish an auditable internal baseline for the current `mamba` profile before changing the architecture. This protocol does not define a clinical acceptance threshold and does not establish clinical performance.

## Freeze the experiment inputs

If the intended source is hosted on Hugging Face, first run `inspect_hf_dataset.py` as described in the README and retain its JSON report. Pin a dataset revision, confirm the image and label columns, map labels to the model's class order, inspect the source's license, and verify whether candidate patient/procedure/video fields are stable IDs. The preflight tool is informational; it does not adapt the training loader or establish that a split is leakage-free.

Before training, record:

1. The exact Kvasir-v2 source/version and local root directory.
2. Whether patient, procedure, video, or source metadata exists. If stable pseudonymous patient/procedure/video IDs are available, provide `groups.csv` (`relative_path,group_id`) at the dataset root. The split builder will keep all mapped frames together and also link near-duplicates across groups. Without this metadata it falls back to perceptual near-duplicate grouping, which cannot establish patient/procedure independence.
3. The intended profile: `mamba`, unless the experiment explicitly names another profile.
4. Seed and all training arguments. Start with seed `42` and record every argument. The current training defaults include five warmup epochs and MixUp/CutMix; to reproduce the earlier recipe before these additions, use `--warmup-epochs 0 --mixup-alpha 0 --cutmix-alpha 0` in a separate run directory. Do not change profile or hyperparameters midway through a run.

## Run sequence

From the project directory and the activated environment:

For a college Linux/NVIDIA server, use the dedicated setup and commands in the README's **College NVIDIA GPU training** section. Run inside the GPU allocation and preserve the environment and GPU detail files with the run artifacts.

```powershell
python smoke.py
python train.py `
  --data-root "D:\datasets\kvasir_v2" `
  --output-dir ".\runs\baseline_mamba_seed42" `
  --profile mamba `
  --seed 42
python evaluate.py `
  --data-root "D:\datasets\kvasir_v2" `
  --checkpoint ".\runs\baseline_mamba_seed42\best_model.pt" `
  --run-dir ".\runs\baseline_mamba_seed42" `
  --output ".\runs\baseline_mamba_seed42\test_metrics.json"
```

Replace the example data path with the actual dataset directory. The training loader validates the eight expected class folders, groups patient/procedure/video frames when `groups.csv` is present and near-duplicate images in all cases, persists the split and metadata hash, and selects a checkpoint using validation macro-F1. It validates the full saved ratio configuration, split membership, stored image counts, and per-class counts; it stops if any class is absent from train, validation, or test. The test split is only evaluated after training. `evaluate.py` verifies the dataset content, split file, run manifest, final training result, selected epoch, and checkpoint hash before it reports the internal test result.

Each training invocation requires a new or empty `--output-dir`. The trainer refuses to write into a populated run folder, protecting checkpoints, splits, manifests, and reports from accidental overwrite. Use a new directory for every repeat, recipe, seed, or resume attempt. The current trainer does not resume interrupted runs.

When comparing the previous run's reported 85% accuracy with new training methods, use the same dataset content, class mapping, profile, and fixed split. Run the previous recipe and each proposed recipe into separate output directories; compare validation macro-F1 and per-class recall to choose a candidate. Keep the test split locked during selection, then evaluate the selected candidate once. The new MixUp/CutMix and warmup options do not alter the network or `infer_video.py`; no gain should be claimed until a completed held-out evaluation report supports it.

## Required evidence to retain

Keep the entire output directory together:

- `splits.json` — fixed file assignment and split metadata.
- `model_config.json` — selected architecture and class order.
- `run_manifest.json` — command, arguments, package/runtime versions, CPU and accelerator model details, memory/capability, CUDA/cuDNN and thread settings, aggregate and per-image dataset fingerprints, source fingerprint, and split hash.
- `history.jsonl` — epoch-level training and validation history.
- `best_model.pt` — selected checkpoint and its provenance reference.
- `training_result.json` — best epoch and checkpoint hash.
- `test_metrics.json` — held-out metrics, confusion matrix, and provenance checks.
- `test_metrics_predictions.csv` — one row per held-out image with the true/predicted classes and all class probabilities, for false-positive/false-negative review.

Before describing the result, verify the run completed, the provenance checks pass, all class supports are reported in `splits.json` and `test_metrics.json`, the confusion matrix sums to the number of test examples, and the prediction CSV contains exactly one row per test image. Review false positives and false negatives by class using the CSV and record representative examples. Include the raw multiclass Brier score and top-label 15-bin ECE as diagnostics; do not describe these raw scores as calibrated or use them for abstention without a separately fitted and validated calibration policy. Report the exact split, seed, model profile, environment, metrics, and known data limitations together. Do not claim a target accuracy unless it is measured on the completed test report and its evaluation protocol is stated.

## Independent external evaluation

Only after the internal baseline is frozen, use a separately sourced cohort with an agreed one-to-one mapping to the eight trained classes:

```powershell
python evaluate_external.py `
  --data-root "D:\datasets\external_kvasir_labels" `
  --dataset-id "source-name/version/site-date" `
  --confirm-independent `
  --checkpoint ".\runs\baseline_mamba_seed42\best_model.pt" `
  --output ".\runs\baseline_mamba_seed42\external_metrics.json"
```

This scores the full directory and does not create/reuse an internal split. It checks for byte-identical images against the per-image digests in the training manifest and stops if it finds overlap. The command's `--confirm-independent` flag is an operator attestation, not automated proof of source or patient/procedure independence. Exact-byte matching will not find altered or visually near-duplicate frames. Record the external source, label mapping, inclusion rules, patient/procedure grouping, authorization, and any overlap checks alongside the JSON report. Do not use the external cohort for model selection or threshold tuning if its results are to be reported as independent validation.

## Held-out video evaluation

For an independent labeled clip cohort, arrange one-class-per-clip videos in eight class folders and run `evaluate_video.py` after freezing the model. It reports per-frame metrics, clip predictions from mean sampled-frame probabilities, class-switch rate, and throughput. The default samples every fifth decoded frame; record any change to `--sample-stride`. The cohort must not have been used for training, calibration fitting, or model selection. This tool evaluates the current frame-based classifier and is not a temporal model or proof of patient-level independence.

```powershell
python evaluate_video.py `
  --data-root "D:\datasets\held_out_videos" `
  --dataset-id "source/version/site-date" `
  --confirm-independent `
  --checkpoint ".\runs\final\best_model.pt" `
  --calibration ".\runs\final\confidence_calibration.json" `
  --sample-stride 5 `
  --output ".\runs\final\video_metrics.json"
```

## Known limitations to carry into the baseline report

- The current dataset interface expects eight Kvasir class folders and uses image-level class labels.
- The original class-folder data may not supply pixel-level lesion masks or patient/procedure groups. The optional `groups.csv` only helps when trustworthy pseudonymous identifiers are available.
- Near-duplicate exclusion reduces one leakage risk; patient/video-level independence is only represented when valid group metadata is supplied.
- This baseline measures the selected classifier on an internal held-out split; it is not external validation or clinical validation.
- The model's heatmaps and heuristic evidence box are not ground-truth lesion localization.

## Current prerequisites / blockers

The available local Python runtime currently lacks PyTorch and torchvision, and the dataset directory has not yet been identified. A `best_model.pth` candidate exists outside the project, but its checkpoint keys/configuration do not match this implementation's expected checkpoint schema; do not treat it as a baseline or load it into this model until its original architecture/configuration and evaluation report are obtained. Do not run the training command until a compatible environment and exact data root are available.
