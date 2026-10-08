# Financial Fraud Detection under Extreme Class Imbalance

*A Leakage-Free Comparative Study of Classical Machine Learning and Tabular Transformers*

MSc dissertation by **Olavo Miguel Cabaço Caixeiro**, Universidade Politécnica de
Santarém, supervised by **Maryam Abbasi** and **Pedro Miguel de Oliveira Martins**.

[Read the dissertation](publications/thesis.pdf)
· [Read the manuscripts](publications/README.md#companion-manuscripts)
· [Inspect the results](results/) · [Explore the code](src/)
· [Obtain the datasets](datasets/)

This release contains the final October 2026 dissertation export (V6), the
matching selected scientific evidence and two reviewed companion manuscripts
in preparation for submission. The manuscripts are not yet published.
This is research code, not a production fraud system, a journal-acceptance
claim or confirmation of an awarded degree.

This original repository retains the development history and is the main
repository for the dissertation. It was renamed from `FraudDetection` on
8 October 2026 without replacing its history. Dedicated code and evidence
packages accompany the manuscripts:

- [Article 1: imbalance sensitivity, transfer and interpretability](https://github.com/Applied-Intelligence-Hub/transformers-vs-ml-fraud-detection).
- [Article 2: threshold selection and alert workload](https://github.com/Applied-Intelligence-Hub/tabular-ml-dl-threshold-benchmark).

The manuscripts share experimental evidence; they are not independent
replications. The PDF exports currently supplied precede the final repository-link
updates to the LaTeX sources and will be replaced after author-side recompilation.

## Study and principal findings

The study separates **ranking**, **validation-selected decision thresholds**
and **alert workload**. Five supervised model families and an OCSVM reference
are evaluated on ULB credit-card transactions and BAF bank-account applications.

| Baseline model | ULB PR-AUC | BAF Base PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6921 | 0.1432 |
| Random Forest | 0.7928 | 0.1588 |
| LightGBM | 0.8205 | 0.1768 |
| CatBoost | 0.8196 | 0.1796 |
| FT-Transformer | 0.8015 | 0.1769 |
| One-Class SVM | 0.2593 | 0.0190 |

PR-AUC is average precision. Values are descriptive point estimates, not paired
tests of between-model superiority. No single model dominates every criterion.

- **Thresholds matter:** BAF LightGBM TEST F2 rises from 0.042 at threshold 0.5
  to 0.316 with the DEV-selected max-F2 rule. Ranking scores do not change;
  recall and review workload do.
- **Imbalance effects depend on the pipeline:** BAF FT-Transformer AP declines
  from 0.1769 to 0.1149 under primary SMOTE (about 35%). The predefined SMOTE-NC
  control reaches 0.1157, not the baseline. Representation differences prevent
  an architecture-only causal interpretation.
- **Transfer is not determined by the in-domain ranking:** after exact profile
  overlap removal, LightGBM leads Variant I AP and CatBoost leads II-V. Models,
  preprocessors and Base thresholds remain frozen.
- **Interpretability has boundaries:** compatible LR/LGBM/CatBoost SHAP results
  concern feature-set agreement, not causality. Final-layer FT-Transformer
  attention is diffuse on average (normalised entropy 0.985), not a faithful
  attribution method or evidence of a feature-space performance ceiling.
- **SMOTE and SMOTE-Tomek can legitimately coincide:** the inspected sampler
  inputs have zero Tomek removals. Equality is supported by sampler and saved
  prediction evidence, not merely rounded table cells.

See the [ULB baseline](results/tables/ulb/baseline.tex),
[BAF baseline](results/tables/baf/baseline.tex),
[threshold sensitivity](results/tables/baf/threshold_f2.tex),
[transfer](results/tables/baf/crossdomain_prauc.tex) and the complete discussion
in the [dissertation](publications/thesis.pdf).

## Experimental protocol

| Component | Procedure |
|---|---|
| ULB | Raw 284,807 transactions / 492 frauds; exact predictor-profile deduplication before splitting retains 283,726 profiles / 473 frauds. TEST contains 56,746 observations / 95 frauds. |
| BAF | Base and five controlled million-row variants; semantic numerical and nominal inputs. |
| Outer split | Stratified 80% DEV / 20% TEST, seed 42; transformations fitted within training partitions. |
| Baseline selection | 50 Optuna trials: five-fold inner validation for classical models; one internal DEV holdout for FT-Transformer. ULB searches were repeated; original BAF-selected parameters were retained. |
| Primary factorial | Five supervised models x seven strategies x two datasets, plus two separate OCSVM references: 72 primary cells. Baseline-selected parameters are reused across interventions. |
| Threshold study | Fixed 0.5, max-F1, max-F2 and the lowest feasible validation threshold with precision at least 0.5, applied to the same saved baseline TEST scores. |
| BAF transfer | Common 30 predictors; Variant TEST profiles matching any Base DEV profile are removed. Retained cohorts have 177,526-178,828 rows. No Variant-specific fitting or threshold selection. |
| Absence codes | Documented numerical sentinels are preserved in the primary BAF pipeline; separate training-fitted NaN/mean-imputation/indicator controls are reported. |
| Categorical sampling | Separate predefined CatBoost/FT SMOTE-NC controls complement the primary representation-dependent SMOTE path. |
| Uncertainty | 1,000 TEST-row bootstrap resamples conditional on fitted models and frozen thresholds; paired intervals for predefined controls. |

Exact-profile separation does not establish entity-level independence or
prospective temporal performance. BAF months are pooled; `x1`/`x2` from Variants
III/V are omitted under the common-feature transfer protocol. Cohorts can overlap
with Base TEST or one another. ULB estimates concern distinct predictor profiles,
not the original transaction-frequency population.

Single splits and conditional bootstrap intervals do not measure training-seed
or tuning variability. Overlapping marginal intervals do not establish
equivalence. The precision constraint applies to validation, not automatically
to TEST or deployment. Recorded computational costs reflect heterogeneous
historical/current workflows and CPU/GPU use, not intrinsic speed-up factors.

## Repository contents

```text
datasets/       Official-source instructions and exact raw-file hashes
notebooks/      Exploratory data analysis
src/            Scientific pipeline, diagnostics and evidence checks
tests/          Synthetic protocol and integrity tests
results/        Final selected metrics, tables, figures and provenance
publications/   Reviewed dissertation and companion manuscript PDFs
```

[results/README.md](results/README.md) explains the evidence export.
Configurations and source identities distinguish corrected runs from preserved
BAF artefacts. The complete local archive is retained by the author, but model
weights, row-level scores/attributions, operational logs and writing projects
are not distributed in this compact publication. Those full artefacts are
necessary to recompute exact historical fitted outputs; published summaries
alone are not a bitwise reproduction package.

`.gitignore` protects local datasets, training outputs and writing materials.
`.gitattributes` preserves Git LFS rules for large binary artefacts if they are
versioned again. The current tree needs no raw dataset or model LFS download
to read the thesis and inspect the exported evidence. Previous commits of this
existing repository still contain historical files; no history rewrite has
been performed.

## Getting started

Use Python 3.12. The recorded correction environment was Python 3.12.14 on
Windows 11, an Intel Core i5-13600KF, 16 GB RAM and an NVIDIA RTX 3070 (8 GB).
These are recorded conditions, not certified minimum specifications.

```powershell
git clone https://github.com/Olavo200100274/fraud-imbalance-classical-vs-transformers.git
cd fraud-imbalance-classical-vs-transformers
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The requirements record the CUDA 12.1 PyTorch 2.5.1 build. A CPU-only alternative
can retain the other pins without installing that CUDA wheel:

```powershell
$projectPackages = Get-Content requirements.txt | Where-Object { $_.Trim() -and $_ -notmatch '^(--|torch==)' }
python -m pip install $projectPackages
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Choose one route; do not reinstall the original CUDA requirements over the CPU
route. See [official PyTorch versioned instructions](https://pytorch.org/get-started/previous-versions/#v251).
Fresh installation and execution on another computer have not been tested for
this publication. Identical results across hardware or dependency changes are
not guaranteed.

The UTF-8 setting prevents Windows console encoding errors in scientific-symbol
progress messages; it does not change any model or metric computation.

### Data and public evidence checks

Obtain CSVs from the original providers using [datasets/README.md](datasets/README.md),
then verify their exact hashes:

```powershell
python src/verify_datasets.py
```

Inspect/check the compact public evidence without training or downloading data:

```powershell
python src/export_public_evidence.py --verify
```

The verification checks publication hashes and internal numerical consistency;
it is not an independent replication of training.

### Isolated fresh training

Always use a **new output directory and explicit manifest**. Never train into
the frozen public `results/` tree. This small pipeline check is not a thesis result:

```powershell
python src/main.py --dataset ulb --models logreg --strategy none --sample 0.05 --n_trials 1 --bootstrap-iterations 0 --results-root runs/smoke --run-manifest runs/smoke/revision_manifest.json
```

For a full independent rerun, fit baselines first, then interventions using the
same new study manifest:

```powershell
python src/main.py --dataset ulb --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset ulb --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
```

Classical `--models all` includes OCSVM, not FT-Transformer. `--strategy all`
means the six non-baseline strategies. Full searches may take hours. The final
dissertation retained verified historical BAF classical models and selected
hyperparameters; fresh searches are an independent rerun, not an exact replay
of those preserved runs.

Derive the baseline threshold study from saved validation scores and perform
frozen-model BAF transfer:

```powershell
python src/threshold_study.py --dataset ulb --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/ulb_2013
python src/threshold_study.py --dataset baf_base --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/baf_base
python src/revision_transfer.py --manifest runs/reproduction/revision_manifest.json --models all --output-dir runs/reproduction/derived/transfer --bootstrap-iterations 1000 --device cpu
```

Compatible SHAP, attention and the predefined controls require their explicitly
pinned sources. The specialised October queues assume the full author-side
archive and must not be launched as an unconfigured generic replication recipe.
See [src/README.md](src/README.md). The reporting generator refuses incomplete
primary/control/interpretability evidence; it does not invent absent results.

Synthetic protocol tests (no full-data retraining):

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Citation and rights

> Caixeiro, Olavo Miguel Cabaço (2026). *Financial Fraud Detection under Extreme
> Class Imbalance: A Leakage-Free Comparative Study of Classical Machine Learning
> and Tabular Transformers*. MSc dissertation, Universidade Politécnica de Santarém.

Include the repository URL and commit used when discussing code or results.
Cite dataset authors and third-party methods separately; the dissertation
contains the bibliography. No project-wide open-source licence is currently
declared. Dataset, institutional-asset and third-party rights remain with their
respective owners.
