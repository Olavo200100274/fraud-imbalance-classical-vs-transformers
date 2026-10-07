# Dataset inputs

Raw CSVs are deliberately excluded from the current public tree. Obtain them from
the dataset authors and place them here; no dataset needs to be downloaded merely
to read the dissertation or inspect the published results.

| Dataset | Official source | Required local filename |
|---|---|---|
| ULB Credit Card Fraud Detection | [ULB on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard_2013.csv` (rename `creditcard.csv`) |
| Bank Account Fraud suite | [Feedzai documentation](https://github.com/feedzai/bank-account-fraud) and its [Kaggle download](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | `Base.csv`, `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, `Variant V.csv` |

Download and decompress the CSV versions. Preserve their contents: changing
column order, line endings or numeric formatting changes the file hash. Follow
the original providers' access, attribution and licence terms; this repository
does not grant a new licence for their datasets.

[manifest.json](manifest.json) records the byte sizes and SHA-256 hashes of the
seven exact files used in the study. Verify downloaded inputs from the project
root with:

```powershell
python src/verify_datasets.py
```

For a ULB-only check:

```powershell
python src/verify_datasets.py --files creditcard_2013.csv
```

The loader removes exact ULB predictor-profile repetitions before splitting.
BAF uses the documented pooled-month split and the common 30-predictor projection;
the `month` column is not a model input, and `x1`/`x2` are excluded from Variants
III/V for transfer. These are experiment procedures, not edits to the raw CSVs.

Previous commits in this existing repository contain dataset LFS pointers. The
current publication does not require those historical objects; a future clean
repository will not carry that history.
