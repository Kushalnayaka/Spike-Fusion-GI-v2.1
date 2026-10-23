# SpikeFusion-GI Customer Requirements Gap Analysis

**Basis:** `SpikeFusion_GI_Customer_Requirements_v2_Clean.pdf` (Customer Requirements Specification v2.0) and the current project source.
**Review date:** 2026-10-23

## How to read this review

The supplied PDF labels itself a *simulated* hospital/endoscopy-center customer review. The user has adopted it as the project's working target. Its statements are requirements and acceptance expectations, not evidence that a hospital has approved the product, that the model is clinically validated, or that regulatory requirements have been satisfied.

Status labels distinguish source-code support from demonstrated evidence:

- **Partial — code exists, evidence pending:** a prototype capability is present, but the requested result has not been measured or independently validated.
- **Missing:** no implementation or evidence was found in the current project.
- **Human/external dependency:** requires dataset access, target hardware, clinical collaborators, or qualified regulatory/security review.

## Current baseline

The project contains an eight-class Kvasir image classifier with SNN and CNN branches, gated fusion, and selectable sequence profiles. It includes duplicate-aware split persistence, seeded training, validation checkpoint selection, held-out test evaluation, image-level Grad-CAM/SpikeCAM/SHAP tooling, a per-frame webcam/video prototype, and TorchScript/optional ONNX export. Training has optional MixUp/CutMix batch augmentation and linear learning-rate warmup before cosine decay; the student model and video inference interface are unchanged. These are experiment options only, and no accuracy benefit has been measured. Baseline run manifests fingerprint code, exact image content, split files, checkpoints, environment, and training arguments; the internal evaluator verifies these identities for provenance-enabled checkpoints. `train.py` now refuses to write into a non-empty output directory, reducing accidental replacement of prior run evidence. The new `inspect_hf_dataset.py` reports Hugging Face split/schema/label-count metadata to guide dataset adaptation; it does not load data into the trainer or establish patient-level leakage control.

These are code capabilities only. No trained checkpoint, test report, external validation result, calibration result, sustained video benchmark, or target-device benchmark has been produced in this environment. The video script currently classifies frames independently; its optional probability averaging is not temporal reasoning. The image evidence box is derived from attribution heatmaps and is not lesion segmentation. The data split uses perceptual duplicate grouping; it does not group by patient or procedure metadata.

## P0 requirements — mandatory before a serious clinical pilot

| ID | Requirement | Current status | Evidence needed / gap |
|---|---|---|---|
| TR-P0-01 | Reproducible baseline | **Partial — provenance and run-overwrite guard implemented, run evidence pending** | Training records arguments, environment/package versions, source fingerprint, aggregate and per-image content fingerprints, split hash, checkpoint hash, and test-report provenance. `train.py` refuses to write into a populated output directory, keeping repeated experiments in separate folders. `evaluate.py` writes a per-image prediction CSV for false-positive/false-negative review. Still run and archive a baseline release with a real checkpoint, held-out report, confusion matrix, and reviewed failure cases before adding model complexity. |
| TR-P0-02 | External validation | **Partial — full-dataset evaluator and exact-image overlap gate added; independent evidence pending** | `evaluate_external.py` scores every image in a separately named eight-class dataset, fingerprints it, rejects exact byte-identical images found in the training cohort, and records an operator attestation. Exact-file matching cannot prove patient/procedure independence or label equivalence. Run it on a clinically appropriate untouched cohort and report per-class metrics and the gap from the frozen internal baseline. |
| TR-P0-03 | Leakage control | **Partial — optional group-aware splitting added; dataset metadata/evidence pending** | Add a root `groups.csv` with `relative_path,group_id` to keep all frames from the same pseudonymous patient/procedure/video together; the splitter also links near-duplicates and fingerprints the CSV. Without valid IDs, only near-duplicate grouping and external exact-image checks apply. Patient/procedure metadata and cross-source near-duplicate review remain external data requirements. |
| TR-P0-04 | Temporal intelligence | **Missing — held-out video evaluation tooling added; temporal model still absent** | `evaluate_video.py` reports frame metrics, mean-probability clip metrics, class-switch rate, and throughput on labeled independent videos. This measures the current frame classifier and does not create temporal representations, lesion tracking, or event-level reasoning. A sequence-aware method and procedure-level validation remain future work. |
| TR-P0-05 | Calibrated uncertainty | **Partial — temperature-scaling workflow added; evidence and abstention policy pending** | `calibrate.py` fits a scalar temperature only on the saved validation split and binds the calibration JSON to the exact checkpoint and class order. Evaluation can report raw and temperature-scaled metrics, and image/video inference can display scaled scores. This does not change accuracy or class decisions. Measure Brier/ECE on the held-out test set and an independent cohort; no abstention threshold or coverage-risk evidence exists. |
| TR-P0-06 | Unknown/OOD handling | **Missing** | Define held-out unknown distributions and evaluate explicit unknown/abstain behavior. |
| TR-P0-07 | Image-quality gate | **Missing** | Identify poor-quality frames and test whether outputs are suppressed or downgraded safely. |
| TR-P0-08 | Fail-safe operation | **Partial — local display fallback added; system integration missing** | `infer_video.py` suppresses failed/non-finite predictions, clears stale smoothing history, displays an AI-unavailable state, and continues the local preview. It does not prove the endoscope primary-video path is isolated or uninterrupted; fault injection and target-system integration evidence are still required. |
| TR-P0-09 | Edge benchmark | **Missing / external dependency** | Measure end-to-end latency, sustained FPS, memory, dropped frames, power, and thermal behavior on the named target hardware. |
| TR-P0-10 | Clinical validation plan | **Partial — draft prepared / human review required** | `CLINICAL_VALIDATION_PLAN_DRAFT.md` lists staged studies, reference-standard decisions, outcomes, and open inputs. Clinical/statistical collaborators must define and approve the population, endpoints, sample size, protocol, and responsibilities. |
| TR-P0-11 | Regulatory pathway | **Partial — inventory only / qualified assessment missing** | `RISK_REGISTER_DRAFT.md` and the validation draft identify intended-use and safety questions. A qualified regulatory/quality professional must perform the product-specific pathway assessment. |

## P1 product capabilities

| ID | Requirement | Current status | Evidence needed / gap |
|---|---|---|---|
| TR-P1-01 | Lesion localization | **Partial** | Still-image CAM heatmaps and a heuristic evidence box exist; video can optionally generate the same attribution overlay every N frames. These are not validated spatial outputs or lesion boundaries; obtain expert masks and evaluate localization/segmentation if this becomes a product claim. |
| TR-P1-02 | Persistent lesion tracking | **Missing** | No lesion identity across frames or views. |
| TR-P1-03 | Multi-finding handling | **Missing** | Current head produces one image-level class prediction, not multiple localized findings. |
| TR-P1-04 | Procedure timeline | **Missing** | No timestamped, deduplicated procedure event record. |
| TR-P1-05 | Model/version registry | **Missing** | No release registry, provenance record, signed artifact, or rollback workflow. |
| TR-P1-06 | Audit trail | **Missing** | No persistent log of events, model version, errors, overrides, or configuration changes. |
| TR-P1-07 | Hardware portability | **Partial** | Export paths exist; runtime/operator support and performance are not verified across devices. |
| TR-P1-08 | Compression study | **Missing** | Dynamic INT8 for linear layers is an option, not an FP32/FP16/INT8 comparison with accuracy-speed-memory results. |
| TR-P1-09 | Offline-first | **Partial** | Local model inference is possible; offline behavior has not been tested in an integrated system. |
| TR-P1-10 | Robustness | **Missing** | No measured tests for blur, illumination, compression, motion, capture variation, or domain shift. |

## Product and organization gaps

- **Backend:** no inference service/API, database, model registry, or event store.
- **Frontend:** the current video tool is a developer display, not a clinician-facing UI; it has no workflow controls, AI-unavailable state, or review timeline.
- **MLOps/DevOps:** no release pipeline, monitoring, signed/versioned model package, controlled update, or rollback procedure.
- **Clinical:** validation plan is a draft; no participating gastroenterologist, approved annotation protocol, or clinical workflow review is recorded in the project.
- **Quality/regulatory/security:** an initial unscored risk inventory and candidate intended-use wording exist as drafts; no approved risk analysis, software lifecycle records, privacy/security controls, or qualified regulatory assessment exist.
- **Data rights:** the README now records the Kvasir provider's research/education restriction and commercial written-permission requirement; no project-specific permission or dataset use approval is recorded.
- **Product/business:** no documented pilot scope, customer discovery, pricing, support model, or measured ROI hypothesis.

These gaps require real clinical, regulatory, security, and customer collaborators. Engineering work can prepare artifacts and evidence for their review; it cannot substitute for their decisions or sign-off.

## Recommended staged plan

1. **Baseline first:** establish the environment and dataset version; run model smoke checks; produce a reproducible training run, fixed split, checkpoint, held-out test metrics, confusion matrix, and failure-case examples. Freeze this as the comparison baseline.
2. **Generalization and leakage:** obtain an independent dataset; use patient/procedure grouping when metadata permits; compare external versus internal results.
3. **Temporal representation:** collect or secure procedure-level clips and labels; compare frame-only inference with sequence-aware inference using event-level measures.
4. **Reliability and safety:** add calibration, abstention/OOD handling, image-quality gating, and a fail-safe video path; evaluate each on predefined held-out conditions.
5. **Spatial/event capability:** only with appropriate annotations, evaluate localization; then add persistent tracking, multi-finding handling, and procedure timelines.
6. **Deployment evidence:** choose the actual target device; compare FP32/FP16/INT8; measure sustained performance, resource use, drops, thermals, and failure recovery.
7. **Pilot readiness:** in parallel from the start, define intended use with clinical input, prepare the validation protocol and risk documentation, and obtain qualified regulatory/privacy/security review.

Every model/system change should be compared with the frozen baseline and released with its dataset/split provenance, model version, validation evidence, limitations, and rollback path.

## Immediate blocker for Stage 1

The available Python 3.12 runtime has neither `torch` nor `torchvision`. The project folder currently has no training dataset, checkpoint, fixed split, or completed metrics report. A separate `best_model.pth` file was found in Downloads, but its root format is `model_state_dict` plus `epoch`, its state keys use `snn_branch`/`cnn_branch`/`mamba` names rather than the current model's `snn`/`cnn`/`sequence` hierarchy, and no matching configuration or evaluation report was found alongside it. Current inference expects `model_state` and `model_config`; therefore this candidate is not a drop-in checkpoint and is not accepted as baseline evidence. To assess/reuse it, obtain its exact source/configuration and validation report, then perform a controlled compatibility check. To produce a baseline for this project, the team still needs a compatible PyTorch environment and the exact dataset directory/version. Patient/procedure grouping requires valid optional `groups.csv` metadata; the class-folder dataset does not provide those identifiers or pixel-level lesion masks on its own.
