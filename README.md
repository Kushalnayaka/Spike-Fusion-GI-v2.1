# SpikeFusion-GI v2.1 — reproducible implementation

This is a new, standalone implementation based on the public [Spike-Fusion-GI-v2.1 repository](https://github.com/Kushalnayaka/Spike-Fusion-GI-v2.1). It does not modify that repository or the pre-existing loose prototype files. The 95–97% accuracy range is an experimental aspiration from the upstream README, not a measured or guaranteed result. No dataset training was run while preparing this project.

## What the upstream source actually contains

The upstream README describes a 10-channel retinal encoder, four-step LIF SNN, ECA CNN, gated fusion, three bidirectional Mamba blocks, eight classes, Grad-CAM/SpikeCAM, SWA, warmup/cosine scheduling, and expected 94–97% accuracy. In the inspected `models/spikefusion.py`, the top-level class instantiates the named branch classes, averages sequence outputs, and returns auxiliary tensors; the file alone does not establish that the imported branch internals or XAI work as claimed. `train.py` contains Mixup/CutMix paths, an SWA object and scheduler calls, as well as validation checkpoint selection followed by a test evaluation after training. It uses the CUDA AMP autocast/scaler API without selecting a device-aware CPU path, and the visible loop does not save/use the SWA weights as its best checkpoint. The README labels its metric ranges as expected estimates; they are not measured results. These are source-level observations, not claims about files that were not inspected.

This implementation makes the encoder, SNN, CNN, fusion, sequence block, metrics, split persistence, inference maps and export commands executable and inspectable. The sequence model is a compact Mamba-style selective recurrence written in ordinary PyTorch. It is not the upstream `mamba-ssm` CUDA implementation; installation/build compatibility therefore does not depend on `causal-conv1d`. The `edge` profile removes the recurrence, and the new `micro` profile also narrows the CNN/SNN and token mixer for tighter edge budgets.

## Adopted customer requirements and readiness

The team is using the supplied simulated customer requirements specification as a staged product target. It is not hospital approval or clinical validation. The source-to-requirements audit is in [`REQUIREMENTS_GAP_ANALYSIS.md`](REQUIREMENTS_GAP_ANALYSIS.md); the baseline run procedure is in [`BASELINE_PROTOCOL.md`](BASELINE_PROTOCOL.md); the clinical validation protocol and initial risk inventory are drafts for qualified human review in [`CLINICAL_VALIDATION_PLAN_DRAFT.md`](CLINICAL_VALIDATION_PLAN_DRAFT.md) and [`RISK_REGISTER_DRAFT.md`](RISK_REGISTER_DRAFT.md). Current code capabilities must not be described as measured or clinically validated results.

## Model and parameter budget

Input is RGB float `[B,3,224,224]` scaled to `[0,1]`. Fixed retinal kernels produce 10 channels: two DoG scales, four Gabor orientations, two color-opponency channels, and horizontal/vertical Sobel. A four-step surrogate-gradient LIF network extracts edge activations. In parallel, a small depthwise MobileNet-style CNN with ECA channel attention extracts color and texture. A 1x1 projection plus squeeze gate combines both 7x7 maps into 128-dimensional spatial tokens. Three bidirectional selective-state blocks model tokens; global mean pooling and a linear head produce the eight Kvasir v2 labels.

Default configuration is `image_size=224`, `token_dim=128`, `ssm_depth=3`, `state_dim=16`, `snn_steps=4`, `cnn_widths=[16,24,32,48]`, and `sequence_mode=mamba`. The layer shapes give 314,738 parameters for the default selective-scan profile, 298,100 for the standard-width `edge` profile, and 25,028 for `micro`; all are trainable. The micro profile uses CNN widths `[8,12,16,24]`, SNN widths `[12,16]` with 16 output channels, 48 token dimensions, one convolutional token-mixer block, and three LIF steps. `python smoke.py` calculates each count from the instantiated model and fails if any exceeds 1,100,000. These counts are architecture counts, not a training result.

## Install

PyTorch 2.14.0 is the latest stable release at implementation time (September 25, 2026). The official torchvision compatibility table pairs torch 2.14 with torchvision 0.29. Use Python 3.10–3.14. Choose the correct CUDA/CPU wheel command for the runtime on the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 torchvision==0.29.0
python -m pip install -r requirements.txt
```

Optional explainability/export packages: `python -m pip install shap onnx onnxscript`.
Optional webcam/video capture: `python -m pip install opencv-python`.

The model/data code uses PyTorch and Pillow directly; torchvision is only imported if transfer learning is requested. The upstream Mamba install guide currently documents a default package install without its CUDA selective-scan extension, with the extension opt-in via a CUDA build; it lists Linux, Python 3.10+, preinstalled PyTorch, and CUDA 11.6+ for the NVIDIA extension. `causal-conv1d` is an optional extra. The upstream guide does not publish a tested Mamba/causal-conv1d compatibility matrix for this project’s exact torch 2.14/torchvision 0.29 environment. This implementation therefore uses no external Mamba or causal-conv1d runtime and avoids that build dependency. If adding those packages independently, follow the current official build instructions after choosing a compatible platform build.

## College NVIDIA GPU training

Run inside an allocated GPU session/job on the college Linux server, not on a login node. Inspect the GPU and driver, create a Python environment in your home or project scratch storage, then install the matching NVIDIA PyTorch wheel:

```bash
nvidia-smi
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to choose the CUDA build supported by the server's driver, pinning PyTorch 2.14.0 and torchvision 0.29.0 for this code snapshot. Then install the project dependencies and verify CUDA is available:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA DEVICE')"
```

Copy this project folder and an authorized dataset copy to storage accessible from the compute node. Set `DATA_ROOT` to the directory containing exactly the eight class folders; do not point it to the outer archive directory or zip file. For the first baseline, keep the `mamba` profile, seed 42, and documented defaults:

```bash
DATA_ROOT=/path/to/kvasir_v2
RUN_DIR=/path/to/runs/baseline_mamba_seed42
set -euo pipefail
mkdir -p "$(dirname "$RUN_DIR")"
if [ -e "$RUN_DIR" ]; then echo "Choose a fresh run directory: $RUN_DIR"; exit 1; fi
python smoke.py
python -m pip freeze > "${RUN_DIR}.pip-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "${RUN_DIR}.gpu.csv"
python train.py --data-root "$DATA_ROOT" --output-dir "$RUN_DIR" \
  --profile mamba --seed 42 --epochs 120 --patience 20 \
  --batch-size 32 --workers 4 2>&1 | tee "${RUN_DIR}.train.log"
python evaluate.py --data-root "$DATA_ROOT" \
  --checkpoint "$RUN_DIR/best_model.pt" --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/test_metrics.json"
```

If the job runs out of GPU memory, retry with a smaller batch size in a **new** run directory and record it as a separate experiment. Keep the complete run directory plus the pip-freeze, GPU, and training-log files. The run directory should contain the manifest, split, checkpoint, training history/result, held-out metrics, and `test_metrics_predictions.csv`; review its false positives and false negatives before reporting results. Kvasir-v2 is research/education-only unless written commercial permission is obtained.

`train.py` refuses to use a populated output directory. Choose a new or empty `--output-dir` for every run; interrupted-run resume is not implemented.

## Colab GPU quick start

1. In Colab select **Runtime → Change runtime type → GPU** and verify `torch.cuda.is_available()`.
2. Clone/copy this standalone folder into the notebook, install the packages above (select the PyTorch CUDA wheel matching the current Colab runtime), then mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Put Kvasir v2 under `/content/drive/MyDrive/datasets/kvasir_v2`. The directory must contain exactly the eight class subfolders below, each with readable images. `train.py` validates class folders and image counts before creating splits.

```text
kvasir_v2/
  dyed-lifted-polyps/  dyed-resection-margins/  esophagitis/
  normal-cecum/  normal-pylorus/  normal-z-line/
  polyps/  ulcerative-colitis/
  groups.csv                 # optional patient/procedure/video grouping metadata
```

For extracted video frames, add a `groups.csv` file with a row for every image and columns `relative_path,group_id`. Give every frame from the same patient the same stable pseudonymous `group_id`; if patient IDs are unavailable, use the procedure ID, then video ID. For example: `polyps/video_01/frame_0001.jpg,case_001`. The split builder keeps both these groups and near-duplicate images in one split and fingerprints the metadata. Do not include names, dates of birth, or other direct identifiers. Without this file, the split can only group near-duplicate images and cannot establish patient/video-level independence.

Kvasir v2 has a public download page at [Simula datasets](https://datasets.simula.no/kvasir/). Its terms restrict use to research and education; commercial use requires prior written permission. Treat a capstone baseline as research/education use only, and do not use the dataset to train or validate a commercial product until the dataset owner has granted written permission. Confirm data storage conditions before copying it to Drive.

## Train and evaluate

### Inspect a Hugging Face dataset before training

The current trainer expects the eight Kvasir class folders shown above. For another Hugging Face dataset, inspect its schema and class distribution before adapting the loader:

```bash
python -m pip install datasets
python inspect_hf_dataset.py --dataset-id owner/dataset-name --revision COMMIT_SHA \
  --output dataset_preflight.json
```

The report lists available split names, candidate image/label columns, per-split label counts, dataset fingerprints, and columns whose names may identify patient/procedure/video groups. Loading may download dataset shards into the Hugging Face cache and use network bandwidth/storage. The tool does not export images or train on the data. Group columns must be reviewed and mapped deliberately; the tool does not infer group semantics. Pin a dataset commit for reproducibility. Use `--trust-remote-code` only after reviewing the repository's dataset code.

```bash
python smoke.py
python train.py --data-root /content/drive/MyDrive/datasets/kvasir_v2 \
  --output-dir /content/drive/MyDrive/spikefusion_run --epochs 120 --patience 20
python evaluate.py --data-root /content/drive/MyDrive/datasets/kvasir_v2 \
  --checkpoint /content/drive/MyDrive/spikefusion_run/best_model.pt \
  --output /content/drive/MyDrive/spikefusion_run/test_metrics.json
```

For a scan-free deployment model, add `--profile edge`. For the smallest model, use `--profile micro`; that profile has about 25K parameters and uses no selective scan. Train and export the same profile so checkpoint tensor shapes match.

### Optional transfer learning

```bash
python train.py --data-root /content/drive/MyDrive/datasets/kvasir_v2 \
  --output-dir /content/drive/MyDrive/spikefusion_micro_tl --profile micro \
  --transfer-learning --transfer-teacher-epochs 10
```

This optional two-stage path uses the ImageNet-pretrained MobileNetV3-Small feature extractor as a training-only teacher, replaces its ImageNet classifier with an eight-class head, warms that head while the feature trunk is frozen, then fine-tunes the trunk. It freezes the teacher afterward and trains the selected compact student with its normal Kvasir cross-entropy loss plus a temperature-scaled KL-divergence loss matching the teacher's eight-class probabilities (`alpha=0.35`, `temperature=3`). The teacher is discarded for deployment; the checkpoint and deployment count include only the student, so the micro deployment model remains around 25K parameters. The teacher weights must be downloadable once or already cached. This is an optional training strategy; no transfer-learning run or accuracy gain has been measured here.

The deterministic split groups visually near-identical images by perceptual hashes before allocating 70/15/15 train/validation/test groups, balancing counts by class. If `groups.csv` is present, those near-duplicate groups are joined with pseudonymous patient/procedure/video groups so no linked frames cross splits. The assignments and metadata hash are saved to `splits.json`; subsequent runs reuse that file and validate that it matches the data and metadata. Without `groups.csv`, the Kvasir folders provide no patient/source grouping, so only near-duplicate grouping is available. The held-out test set is read only by `evaluate.py`, after checkpoint selection on validation macro-F1. Training uses seeded augmentation, device-aware mixed precision, linear learning-rate warmup followed by cosine decay, optional batch-level MixUp/CutMix, early stopping, and best-validation checkpointing. Defaults enable five warmup epochs, MixUp alpha 0.2, CutMix alpha 1.0, and a 0.5 probability of mixing each batch. These training-only methods do not change the student architecture or video inference interface; their accuracy effect must be measured on the fixed validation split.

To reproduce the earlier training recipe without batch mixing or warmup, pass `--mixup-alpha 0 --cutmix-alpha 0 --warmup-epochs 0`. For a controlled experiment, use the same dataset and seed in separate output directories. Select the recipe using validation macro-F1 and per-class results; keep the test split locked until the final recipe is selected, then evaluate it once. An optional `--transfer-learning` run adds a pretrained MobileNetV3-Small teacher and distills its class probabilities into the same student architecture. Treat that as a separate experiment, not a guaranteed accuracy improvement.

Training also writes `run_manifest.json` with the command-line arguments, environment/package versions, hardware, split-file hash, a SHA-256 fingerprint over sorted dataset paths and image bytes, per-image SHA-256 digests, and a source/dependency fingerprint. The selected checkpoint stores a reference to that manifest and the training configuration. `training_result.json` records the selected epoch and checkpoint hash; `evaluate.py` verifies these run artifacts against the checkpoint before adding checkpoint, split, manifest, and evaluated-dataset provenance to its test report. Per-image hashes let `evaluate_external.py` reject exact byte-identical overlap with training images. Computing the dataset fingerprint reads each image file once, adding an initial I/O cost and manifest size in exchange for identifying the exact data used.

Evaluation saves accuracy; macro and micro precision/recall/F1; weighted F1; one-vs-rest macro and micro multiclass ROC-AUC; per-class ROC-AUC; per-class precision/recall/F1/support; multiclass Brier score; top-label ECE using 15 equal-width bins; and the confusion matrix. It also writes `<output-stem>_predictions.csv`, one row per held-out image, with the true/predicted class, top scores, and all class probabilities for reviewing errors. Without a calibration file these diagnostics describe raw softmax scores; the model is uncalibrated and no abstention policy is applied. AUC is `null` for a class if the evaluated labels do not contain both outcomes. Results are only measured after running training and evaluation on the dataset.

### Confidence calibration

After selecting a checkpoint, fit scalar temperature scaling on that run's saved **validation** split, then use the resulting JSON with evaluation and inference:

```bash
python calibrate.py --data-root /path/to/kvasir_v2 \
  --checkpoint runs/final/best_model.pt --run-dir runs/final \
  --output runs/final/confidence_calibration.json
python evaluate.py --data-root /path/to/kvasir_v2 \
  --checkpoint runs/final/best_model.pt --run-dir runs/final \
  --calibration runs/final/confidence_calibration.json \
  --output runs/final/test_metrics.json
python infer.py --image path/to/image.jpg --checkpoint runs/final/best_model.pt \
  --calibration runs/final/confidence_calibration.json --output-dir xai_outputs
```

Temperature scaling changes the probabilities and confidence summaries, but leaves the argmax class and accuracy unchanged. Evaluation reports raw and calibrated metrics separately. The calibration JSON is bound to the exact checkpoint and class order; refit it whenever the checkpoint changes. Calibration can improve confidence reliability but does not prove clinical confidence or provide unknown-class detection/abstention.

### External validation

After an internal baseline is frozen, evaluate a separately sourced cohort with the same eight class folders:

```bash
python evaluate_external.py --data-root /data/external_kvasir_labels \
  --dataset-id "source-name/version/site-date" --confirm-independent \
  --checkpoint runs/baseline_mamba_seed42/best_model.pt \
  --output runs/baseline_mamba_seed42/external_metrics.json
```

This evaluates every image in the supplied external directory without creating or reusing a train/validation/test split. It verifies the checkpoint against its training manifest, split, and result record, rejects exact byte-identical image overlap with the training cohort, then records a content fingerprint, class supports, metrics, and evaluation environment. Exact hashes do not detect altered or visually near-duplicate frames. `--confirm-independent` is an operator attestation; the script cannot prove patient/procedure independence, data-use authorization, or that labels are clinically equivalent. Do not tune the model on this cohort and still call its result external validation.

## Inference and XAI

```bash
python infer.py --image path/to/image.jpg --checkpoint path/to/best_model.pt \
  --calibration runs/final/confidence_calibration.json --output-dir xai_outputs
```

This writes `prediction.json`, `explanation.txt`, `evidence_box.png`, `combined_heatmap.png`, `gradcam.png`, and `spikecam.png`. The red rectangle is a candidate evidence region formed from the largest connected component in the mean Grad-CAM/SpikeCAM map. It marks where the model's attribution is concentrated; it is not a lesion boundary or segmentation mask.

The text/JSON report includes the predicted and runner-up classes, raw or temperature-scaled class scores and margin, whether both attribution maps are usable, their top-region overlap (IoU), prediction stability after a horizontal flip, and an occlusion sensitivity check. That check blurs the candidate box and compares the selected-class probability change with equal-size random boxes. These checks assess model sensitivity and explanation consistency only. A raw softmax score is not calibrated confidence; temperature scaling is only a probability calibration step and is not clinical validation. None of the checks proves that the box contains a lesion or that the prediction is clinically correct. The classification dataset has no pixel-level masks, so precise lesion localization requires a segmentation dataset with expert masks and separate validation.

`--occlusion-samples` controls the random-box comparison count (default 12). Increasing it makes the comparison steadier but adds inference time. Grad-CAM and SpikeCAM are generated together; SHAP remains a separate, optional and slower image-region analysis below.

### Video and webcam inference

After training a checkpoint, classify and display frames from the default webcam:

```bash
python infer_video.py --source 0 --checkpoint path/to/best_model.pt
```

Add `--calibration path/to/confidence_calibration.json` to display validation-temperature-scaled probabilities for that exact checkpoint. Add `--explain-every 30` to compute Grad-CAM/SpikeCAM every 30 frames, overlay their heatmap and a red candidate evidence box, and write a frame-level text explanation to `video_explanations.jsonl`. Explanation frames use an additional gradient pass and may lower display FPS; leave the option at its default `0` for the fastest classification-only stream. The red box is an attribution-derived candidate region, not a validated lesion outline. The still-image `infer.py` workflow already saves the red box, both heatmaps, and a text/JSON explanation report on each image.

```bash
python infer_video.py --source 0 --checkpoint runs/final/best_model.pt \
  --calibration runs/final/confidence_calibration.json \
  --explain-every 30 --output-video runs/final/annotated.mp4
```

Use a camera/capture-card index such as `1`, a video filename, or a stream URL as `--source`. Press `q` or Esc to stop. Add `--output-video processed.mp4` to save the annotated video, and `--smooth-window 5` to average predictions over the last five frames. The default `1` displays each frame's own prediction. OpenCV is optional for the still-image workflow.

This script reads frames, converts each to RGB, center-fits it to 224x224, classifies it, and overlays the top class, score, runner-up, and display FPS. If frame preprocessing or model inference raises an error or returns non-finite probabilities, it clears the rolling score history, labels that frame **AI UNAVAILABLE**, suppresses the prediction, and continues displaying the video. Explanations are optional and run only on selected frames. This is a local prototype fallback only: it does not establish that an endoscope's primary video path remains uninterrupted, and camera/capture-card access, frame rate, codec, and vendor integration still need target-system verification. Predictions remain per-frame results, not a validated real-time clinical system.

### Held-out video evaluation

To measure performance on labeled videos that were not used to train or select the checkpoint, organize one-class-per-clip videos under the same eight class names and run:

```bash
python evaluate_video.py --data-root /data/held_out_videos \
  --dataset-id "source/version/site-date" --confirm-independent \
  --checkpoint runs/final/best_model.pt \
  --calibration runs/final/confidence_calibration.json \
  --sample-stride 5 --output runs/final/video_metrics.json
```

The evaluator reports sampled-frame metrics, mean-probability per-video metrics, class-switch rate, throughput, per-video hashes, and a CSV review file. It requires the operator to attest that the cohort is independent. The video-level result assumes one ground-truth class per clip and averages the sampled frame probabilities. This evaluates the frame classifier on video inputs; it does not add temporal reasoning, tracking, or patient-level independence proof.

For SHAP's model-agnostic image-region explanation, install the optional package with `pip install shap` and run:

```bash
python explain_shap.py --image path/to/image.jpg \
  --checkpoint path/to/best_model.pt --output-dir xai_outputs/shap \
  --max-evals 300
```

This writes a class-specific `shap.png`, raw values in `shap_values.npz`, and metadata in `shap_summary.json`. The image masker replaces hidden regions with a blurred baseline and Partition SHAP estimates each region's contribution to the selected class probability. Attributions depend on the blur baseline and evaluation budget; 300 evaluations can take time. SHAP is an explanation tool and does not change predictions or improve model accuracy.

## Export and edge deployment

```bash
# Try the native selective-scan model (fixed 224x224 spatial input; dynamic batch trace)
python export.py --checkpoint best_model.pt --output-dir export

# Add --try-onnx to attempt ONNX as well
python export.py --checkpoint best_model.pt --output-dir export --try-onnx

# Smaller scan-free standard-width edge profile
python export.py --profile edge --output-dir export-edge

# Smallest scan-free profile (train this profile first for useful predictions)
python export.py --profile micro --output-dir export-micro

# Optional CPU dynamic INT8 quantization of Linear layers
python export.py --checkpoint best_model.pt --output-dir export-int8 --dynamic-int8
```

Export writes a TorchScript traced logits-only model. Pass `--try-onnx` to attempt ONNX; failures are reported without blocking the TorchScript artifact. Fixed spatial size is used because the fusion grid and scan length are configured for 224x224. Batch is trace-dynamic. The recurrence is unrolled over 49 tokens, so ONNX support/performance varies by exporter/runtime and should be validated on target hardware. For a portable deployment, train `edge` or `micro`; both use standard convolution/linear operators and no custom CUDA operators. Dynamic quantization is a CPU inference option and does not quantize convolution weights.

## Source and compatibility references

- [Upstream README and implementation](https://github.com/Kushalnayaka/Spike-Fusion-GI-v2.1)
- [PyTorch 2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/) and [official local installation selector](https://pytorch.org/get-started/locally/)
- [torchvision compatibility table](https://github.com/pytorch/vision#installation) (torch 2.14 ↔ torchvision 0.29)
- [PyTorch `torch.export` documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html) and [ONNX export documentation](https://docs.pytorch.org/docs/stable/onnx.html)
- [Mamba upstream install instructions](https://github.com/state-spaces/mamba#installation)
- [SHAP Image masker](https://shap.readthedocs.io/en/stable/generated/shap.maskers.Image.html) and [image plot example](https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/image.html)
