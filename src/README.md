# Code organisation

The scientific code is retained separately from the private LaTeX projects.

| Area | Entry points / modules |
|---|---|
| Inputs and transformations | `data.py`, `preprocess.py`, `categorical_preprocess.py`, `missing_values.py` |
| Supervised and anomaly baselines | `main.py`, `main_transformer.py`, `models/`, `strategies/`, `evaluation/` |
| Selection and provenance | `experiment_protocol.py`, `ft_run_protocol.py`, `save_load.py` |
| Thresholds and transfer | `threshold_study.py`, `revision_thresholds.py`, `revision_transfer.py`, `cross_domain.py` |
| Compatible explanations | `shap_analysis.py`, `generate_revision_interpretability.py`, `revision_variant_shap.py`, `attention_analysis.py` |
| Prespecified controls and diagnostics | `revision_catboost_control.py`, `revision_paired_analysis.py`, `revision_missing_audit.py`, `revision_sampler_audit.py`, `revision_lgbm_diagnostics.py` |
| Public input/evidence checks | `verify_datasets.py`, `export_public_evidence.py` |
| Tables and figures | `generate_results.py` |

`revision_*` and `run_revision_*` also preserve the provenance checks and
orchestration used for the October correction. These source-specific queues
require the complete local archive and explicit preserved-model pins; they
are not a one-command recipe for reproducing this release from the compact
public export. `revision_thesis_assets.py` is retained because reporting QA
uses its integrity checks; its optional LaTeX-patch mode additionally requires
the private writing project. No private writing sources are shipped here.

The root README gives isolated fresh-training commands. Synthetic checks are
under `tests/` and `src/test_revision_*.py`; they do not retrain the dissertation
grid. Purely editorial PDF review utilities and superseded article plotting
scripts are excluded from the current public tree.
