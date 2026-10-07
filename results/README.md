# Published evidence for the October 2026 dissertation

This folder contains the selected evidence supporting the final dissertation,
not the complete working archive of training jobs or previous thesis versions.

- `metrics/`: the complete 72-cell primary grid, selected configurations and
  numerical source summaries (70 supervised cells and two OCSVM references).
- `tables/` and `figures/`: corrected, evidence-derived publication assets.
- `thresholds/`: four decision rules on each saved baseline's fixed TEST scores.
- `transfer/`: frozen Base-model results on profile-disjoint BAF Variant cohorts.
- `controls/`: predefined absence-code and categorical-aware sampling controls.
- `interpretability/`: compatible SHAP summaries and exploratory attention diagnostics.
- `provenance/`: public relative-path manifest, hashes and original-source identities.

The public export combines corrected runs with explicitly pinned, preserved BAF
artefacts. A historical run being reused does not make superseded ULB or overlapping
Variant evaluations valid. The original source manifests are not rewritten;
public metadata records the export transformations and source hashes separately.

Model weights, complete row-level score/attribution arrays, source snapshots,
operational logs and local editorial audits are intentionally not distributed
in this compact tree. Published summaries support inspection of the reported
evidence, but are not a substitute for these full artefacts when recomputing
the exact historical fitted functions or bootstrap samples. The author retains
the complete working archive locally. New training is an independent rerun,
not a guarantee of bitwise reproduction of preserved models.

Do not train into this folder: it is a frozen reporting export. Use an isolated
ignored output directory such as `runs/reproduction/` and an explicit run manifest.
