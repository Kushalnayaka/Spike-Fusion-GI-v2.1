# SpikeFusion-GI Initial Risk Register - Draft

**Status:** Preliminary engineering inventory for review; not a formal medical-device risk analysis or risk acceptance.
**Date:** 2026-10-03
**Required owners:** Clinical safety lead/gastroenterologist, software/ML lead, edge/integration lead, privacy/security lead, and qualified quality/regulatory professional.

Severity, probability, detectability, residual risk, and acceptability are intentionally **TBD**. Those ratings require the intended use, system architecture, clinical workflow, and qualified risk owners. Do not treat proposed mitigations below as implemented controls unless the verification evidence exists.

| ID | Hazard / failure mode | Potential consequence | Current project state | Proposed risk control | Verification evidence / owner |
|---|---|---|---|---|---|
| R-01 | False negative: a supported finding is assigned another class or missed | Finding may receive less attention if output is over-trusted | Image classifier only; no measured operating point or clinical outcome evidence | Narrow intended use; clinician remains decision-maker; class-specific sensitivity and failure analysis; abstain when uncertainty is high after validation | Held-out/external sensitivity, failure review, human-factors review / Clinical + ML |
| R-02 | False positive or repeated frame predictions create excessive alerts | Alert fatigue, distraction, loss of confidence in system | Video prototype shows per-frame class; optional rolling average; no event deduplication | Sequence-level event logic, alert budget/threshold set with clinicians, one event per tracked finding, reviewable suppression reason | False events per procedure, duplicate rate, usability review / Clinical + Temporal CV |
| R-03 | Uncalibrated softmax interpreted as clinical certainty | Over-reliance on an unreliable score | Reports now include raw multiclass Brier score and top-label ECE; no calibrator or validated abstention exists | Calibrate on separated calibration data; display validated uncertainty state; define abstention policy for intended scope | Brier/ECE/reliability and coverage-risk evaluation on untouched sets / ML + Clinical |
| R-04 | Out-of-distribution image receives a confident supported-class output | Unsupported content may be misrepresented as a known finding | No OOD/unknown mechanism | Define unknown distributions, add an explicit unknown/abstain state, avoid unsupported claims outside the validated domain | OOD false-accept/false-reject tests across sites/devices/classes / ML + Clinical |
| R-05 | Poor-quality frame is classified as if usable | Misleading output during blur, glare, bubbles, darkness, occlusion, or motion | No quality gate | Define a clinical quality rubric and suppress/downgrade outputs for unusable frames | Quality annotations and gate sensitivity/false-suppression results / Clinical + CV |
| R-06 | Grad-CAM/SpikeCAM/evidence box is mistaken for lesion boundary | Incorrect spatial interpretation or missed area | Current maps and box are attribution-based and explicitly labeled non-segmentation | Keep UI language as “model evidence”; do not present as lesion outline; add mask-supervised localization only if claimed | User comprehension test; expert-mask overlap on held-out data / Clinical + CV |
| R-07 | Same lesion appears across many frames, is split/merged, or multiple findings are conflated | Duplicate or missing procedure events | No lesion tracker, multi-finding detector, or procedure timeline | Define identity/deduplication criteria; track uncertainty and lost/reacquired tracks; support multiple instances | Event-level sensitivity, duplicate/split/merge analysis on labeled procedures / CV + Clinical |
| R-08 | Model does not generalize to different sites, devices, populations, or acquisition settings | Degraded or inequitable performance outside benchmark images | Current baseline is Kvasir-v2; external validation not performed | Independent, site/device-stratified validation; prespecified subgroup/error analysis; release-scope limits | External validation report and subgroup coverage / Clinical + ML + Statistician |
| R-09 | Dataset leakage or biased reference labels inflate measured performance | Misleading evidence and unsafe release decisions | Perceptual near-duplicate grouping and exact byte-hash overlap checks exist; patient/procedure metadata unavailable in current folder dataset | Group by patient/procedure/source when possible; record data lineage, label source, and limitations; keep external test untouched | Split audit, metadata review, exact and near-duplicate analysis, annotation/adjudication record / Data + Clinical |
| R-10 | Inference exception, latency spike, or device failure hides AI failure or obstructs primary video | Workflow interruption or silent loss of support | Standalone display now labels a failed/non-finite frame as AI unavailable and continues its preview; endoscope primary-video path remains unintegrated | Run AI as a fail-open sidecar; isolate from primary video; explicit AI unavailable indicator; watchdog/recovery and local logging | Fault-injection tests, long-duration run, dropped-frame and recovery logs / Edge + Integration |
| R-11 | Real-time overload, dropped frames, thermal throttling, or excessive power | Delayed or stale outputs; inconsistent workflow | No exact hardware or sustained benchmark | Benchmark target hardware under procedure-length load; monitor latency, frame age, drops, memory, power, and temperature; define degraded mode | Device benchmark and soak-test report / Edge |
| R-12 | Export/quantization changes predictions or runtime behavior | Unsupported outputs, accuracy loss, or silent incompatibility | TorchScript path, optional ONNX attempt, and linear-only dynamic INT8 option; no target comparison | Version artifacts; compare FP32/FP16/INT8 against baseline; validate ops/device providers and acceptance tolerances | Artifact parity, per-class metric deltas, runtime integration test / Edge + ML |
| R-13 | Wrong checkpoint, class mapping, threshold, or model version is deployed | Outputs do not match validated system | Checkpoint/config fields and run hashes are being added; no registry/rollback system | Signed/versioned release bundle with class order, config, thresholds, code/data provenance, release note, rollback | Deployment checklist, hash verification, rollback drill / MLOps |
| R-14 | Patient identifiers or video leave approved environment or are retained unexpectedly | Privacy/confidentiality breach | No integrated data flow, storage, access, or retention controls | Minimize/avoid patient data; document local/remote flow, access, encryption, retention/deletion, incident response under institutional policy | Security/privacy review and data-flow test / Privacy + Security |
| R-15 | Unauthorized update, vulnerable dependency, or tampered model artifact | Compromised outputs or availability | No update/signing/vulnerability workflow | Controlled updates, dependency inventory, artifact integrity checks, role-based access, rollback, vulnerability response | Security review, update/rollback and access tests / Security + MLOps |
| R-16 | Intended use or interface implies autonomous diagnosis or overstates evidence | Misuse, inappropriate reliance, or regulatory exposure | README labels system unvalidated; no approved intended-use statement | Freeze approved intended use and wording; human-in-loop presentation; regulatory review before external claims | Reviewed labeling/interface and approved scope / Clinical + Regulatory |
| R-17 | Dataset is used outside its permitted license or consent scope | Unauthorized commercial use, contract breach, or invalid data rights | Kvasir-v2 is the planned baseline source; its provider restricts use to research/education and requires prior written permission for commercial use | Record source-specific terms and intended project use; obtain written permission before any commercial training, validation, demonstration, or product use; maintain provenance for every dataset | Retained license/permission evidence and approved data-use record / Product + Legal/Regulatory |

## Immediate containment for the current prototype

- Keep use to engineering/research evaluation; do not use outputs to direct patient care.
- Do not call the candidate evidence box a lesion boundary.
- Use only a trained, versioned checkpoint and retain its run manifest and test report.
- In any video demonstration, make clear that predictions are frame-wise and not temporally validated.
- Keep the endoscope's primary video path independent from the prototype.
- Do not record or transmit patient video until the responsible institution approves the data flow and privacy controls.

## Review record

| Reviewer role | Name / organization | Review date | Decisions / open actions |
|---|---|---|---|
| Practicing gastroenterologist | TBD | TBD | TBD |
| Study statistician | TBD | TBD | TBD |
| Edge / integration owner | TBD | TBD | TBD |
| Privacy / security owner | TBD | TBD | TBD |
| Qualified regulatory / quality professional | TBD | TBD | TBD |
