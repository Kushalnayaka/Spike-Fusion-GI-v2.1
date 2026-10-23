# SpikeFusion-GI Clinical Validation Plan - Draft for Clinical Review

**Document status:** Internal planning draft. Not an approved protocol, clinical recommendation, ethics submission, regulatory assessment, or evidence of clinical performance.
**Version:** 0.1-draft | **Date:** 2026-10-02
**Required reviewers:** Practicing gastroenterologist, biostatistician, study/site lead, privacy/security lead, and qualified regulatory/quality professional.

## 1. Purpose and current product boundary

The purpose of this draft is to plan evidence generation for the current research prototype before any clinical pilot. The current model accepts one RGB frame resized to 224x224 and returns one of eight Kvasir-v2 image-category scores. The video utility applies the classifier to frames independently; optional averaging across recent frame scores is not temporal reasoning. Current Grad-CAM/SpikeCAM maps and the heuristic evidence box are not validated lesion localization.

**Candidate research-only intended use for discussion:** analyze de-identified or otherwise appropriately authorized endoscopy images/video retrospectively, or in a separately approved non-interventional shadow-mode study, to evaluate whether a frozen model can classify the supported Kvasir categories and identify cases requiring further review. No model output should direct patient care under this draft. The clinical advisor and regulatory reviewer must define, revise, or reject this wording before data collection or any prospective study.

The current output classes are: dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, and ulcerative-colitis. The clinician must confirm which of these map to a coherent clinical question and whether the labels/reference standard are suitable for the proposed study. No broader lesion-detection or diagnosis claim is supported by the current output head.

## 2. Questions the study should answer

1. How well does a frozen model reproduce class labels on the internal held-out data and an independent source?
2. Which classes, image conditions, sites, devices, and patient/procedure groups produce errors?
3. Are predicted scores calibrated enough for the narrowly defined research purpose, and can the model safely abstain under pre-specified conditions?
4. Does sequence-aware inference improve procedure-level finding/event performance over independent-frame inference, without unacceptable delay or duplicate events?
5. Does the full video workflow behave safely and remain usable without obstructing the primary clinical video?

## 3. Staged evaluation design

### Stage A - Frozen internal baseline

- Freeze source fingerprint, model profile/configuration, seed, dataset fingerprint, and split file.
- Use the training split for fitting, validation for checkpoint selection, and held-out test split once for baseline reporting.
- Report overall and per-class accuracy-related metrics, confusion matrix, class supports, and representative false-positive/false-negative examples.
- Record dataset provenance and the absence of patient/procedure metadata where applicable. Perceptual-hash grouping alone must not be described as patient-level leakage control.

### Stage B - Independent external evaluation

- Use a separately sourced dataset/site/device cohort that was not used for architecture or threshold selection.
- Predefine label mapping for all eight classes and identify unsupported/ambiguous cases before scoring.
- Preserve patient/procedure/source grouping when metadata permits; report missing grouping metadata as a limitation.
- Report the internal-to-external metric gap overall and per class, with uncertainty intervals and subgroup/error analysis agreed by the clinical/statistical reviewers.
- Keep external test labels sealed from training and tuning. Any subsequent tuning creates a new version and requires a new untouched evaluation set.

### Stage C - Video and temporal evaluation

- Assemble procedure-level clips with timestamps, procedure/patient identifiers where authorized, reference events, and visibility/quality labels.
- Compare the current independent-frame baseline with a sequence-aware method on the same held-out procedures.
- Predefine event-level measures with the clinical advisor: event sensitivity, duplicate-alert rate, false alerts per procedure or unit time, event timing/localization tolerance, and processing delay.
- Report per-frame and per-event results separately. Score smoothing alone is not the temporal model comparator.

### Stage D - Reliability, OOD, and image quality

- On a calibration split separate from test data, assess calibration (including Brier score, expected calibration error with binning details, and reliability plots) and define an abstention policy with clinical input.
- Create a pre-specified unknown/OOD evaluation set representing unsupported categories, devices/sites, and capture conditions. Report detection performance and false-accept/false-reject tradeoffs.
- Label image quality conditions (for example blur, glare, darkness, occlusion, bubbles, and unusable frames) using an agreed rubric. Evaluate quality-gate sensitivity and the rate of unnecessary suppression.
- Test end-to-end fail-safe behavior so AI errors or unavailability do not interrupt the primary video path.

### Stage E - Localization, tracking, and workflow (only if claimed)

- Obtain expert-drawn lesion masks or boxes and an adjudication protocol before training or evaluating localization. Define spatial metrics and matching tolerance in advance.
- Evaluate localization on held-out patients/procedures and independent sources; never use Grad-CAM overlap as a substitute for reference annotations.
- For tracking, define lesion/event identity, duplicate rules, first/last-seen timestamps, missed/split/merged events, and procedure-level scoring with clinicians.

### Stage F - Target-device and workflow study

- Name the exact host device, video source/interface, runtime, model artifact, and software versions.
- Measure end-to-end latency distribution, sustained FPS, dropped frames, memory, power, thermal behavior, startup/recovery, and display delay over representative procedure-length runs.
- Compare FP32, FP16, and INT8 (or target-supported equivalents) with the same evaluation set; report accuracy/reliability changes as well as speed/memory.
- Review user interface, alert burden, AI-unavailable behavior, and workflow with clinicians/technicians before any live use.

## 4. Reference standard and annotations - decisions required

The clinical advisor and study team must specify:

- What constitutes the target finding for each included class and which classes are in scope.
- Source, qualifications, blinding, and number of annotators.
- Whether the reference is an image-level label, procedure/event-level label, localization mask/box, or a combination.
- Adjudication process and handling of uncertain, ungradable, or conflicting labels.
- How pathology, procedure reports, expert review, and/or other evidence contribute to the reference standard.
- Annotation instructions, inter-rater agreement method, and adjudication log.

No reference standard is selected in this draft because the clinical question and use claim are not yet defined.

## 5. Outcomes and statistical analysis - decisions required

Candidate reporting set, to be narrowed and approved before evaluation:

- Per-class support, confusion matrix, sensitivity/recall, specificity, precision, F1, and ROC-AUC where mathematically defined.
- Macro and weighted summaries with confidence intervals; clustered resampling by patient/procedure when those identifiers exist.
- Calibration: Brier score, ECE with binning method, reliability curve, and calibration method fitted only on calibration data.
- Abstention: coverage-risk curve and performance among accepted cases, alongside abstention frequency and class/group breakdown.
- OOD: sensitivity at pre-specified false-positive rates and the distribution of confidence on unknown samples.
- Quality gate: failure-to-detect unacceptable frames and avoidable suppression of usable frames.
- Video: event sensitivity, duplicate/false events per procedure, temporal localization/timing, delay, and dropped-frame measures as agreed with clinicians.
- Localization/tracking: only metrics defined against expert references and relevant to the use claim.

Do not set a sample count, non-inferiority margin, accuracy target, or clinical success threshold until the study population, prevalence, target precision/power, clustering, and intended-use claim are specified with clinical/statistical collaborators. Preserve a written rationale for the final sample size and analysis plan.

## 6. Bias, privacy, and safety controls

- Define the intended population, exclusions, sites, devices, and acquisition variation before evaluation.
- Analyze failures and performance by pre-agreed clinically relevant subgroups when lawful and sample sizes permit; report small or missing groups.
- Use authorized/de-identified data pathways and document access, storage, retention, deletion, encryption, and audit responsibilities with the institution.
- Keep the human clinician as decision-maker in any research workflow; do not allow this prototype to interrupt or replace the primary video.
- Record model version, data/split hashes, thresholds, environment, user-visible outputs, errors, and overrides for each study run.
- Route intended use, risk controls, study procedures, consent/ethics, and regulatory questions to the responsible institution and qualified professionals.

## 7. Release gates

No gate is passed until the named evidence is complete and reviewed:

| Gate | Minimum evidence | Approver(s) to assign |
|---|---|---|
| Internal baseline | Frozen manifest, checkpoint, test report, confusion matrix, and failure review | ML lead + independent reviewer |
| External validation | Independent cohort report, label mapping, and domain-gap analysis | Clinical lead + statistician |
| Temporal/reliability | Procedure-level report, calibration/abstention/OOD/quality results, failure-mode review | Clinical lead + ML/safety leads |
| Localization/tracking | Expert annotations, agreement/adjudication record, spatial/event metrics | Gastroenterologist + CV lead |
| Device/workflow | Exact hardware/runtime benchmark, fail-safe test, workflow review | Edge/IT/biomed + clinical lead |
| Pilot readiness | Approved intended use, protocol, risk controls, privacy/security and regulatory review | Institution + qualified professionals |

## 8. Reporting framework references

Use reporting checklists appropriate to the eventual study design. DECIDE-AI is intended for reporting early-stage live evaluation of AI decision-support systems and emphasizes the whole intervention, human factors, reproducibility, and generalizability. STARD-AI is a reporting guideline for AI-centered diagnostic-accuracy studies. These are reporting aids; they do not replace a study protocol, ethics review, clinical validation, or regulatory assessment.

- Vasey et al., DECIDE-AI, *Nature Medicine* (2022): https://www.nature.com/articles/s41591-022-01772-9
- Sounderajah et al., STARD-AI, *Nature Medicine* (2025): https://www.nature.com/articles/s41591-025-03953-8

## 9. Open decisions

1. Exact intended use and whether the initial activity is retrospective research or a non-interventional shadow-mode study.
2. Gastroenterologist/clinical site and target population.
3. Kvasir-v2 data root/version, permitted research use, and written commercial permission if the data will support commercial model development; identify the external dataset/source and its data-use terms.
4. Which Kvasir classes represent the first clinically coherent target question.
5. Patient/procedure metadata availability and permitted linkage.
6. Reference standard, annotation resources, and sample-size/precision rationale.
7. Target video source, edge hardware, and deployment environment.
8. Qualified regulatory, quality, privacy, cybersecurity, and institutional review owners.
