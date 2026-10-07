"""Count documented BAF absence codes on the explicitly recorded partitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_protocol import array_sha256, file_sha256
from missing_values import BAF_MINUS_ONE_FIELDS, BAF_NEGATIVE_MISSING_FIELDS


def audit(partitions_dir, output):
    partitions_dir = Path(partitions_dir).resolve()
    manifest_path = partitions_dir / "partition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Completed, verified BAF partitions are required.")
    provenance = manifest["base"]["data_provenance"]
    source = Path(provenance["raw_file"])
    if file_sha256(source) != provenance["raw_file_sha256"]:
        raise ValueError("The Base CSV has changed since partition preparation.")
    rows = provenance["eligible_rows"]
    masks = {"raw": np.ones(rows, dtype=bool)}
    for label in ("dev", "test"):
        indices = np.load(partitions_dir / f"base_{label}_row_indices.npy", allow_pickle=False)
        if array_sha256(indices) != provenance[f"{label}_indices_sha256"]:
            raise ValueError("Base partition indices do not reproduce their source hash.")
        masks[label] = np.zeros(rows, dtype=bool)
        masks[label][indices] = True
    if np.any(masks["dev"] & masks["test"]) or not np.all(masks["dev"] | masks["test"]):
        raise ValueError("Base DEV and TEST must form a disjoint complete partition.")
    fields = (*BAF_MINUS_ONE_FIELDS, *BAF_NEGATIVE_MISSING_FIELDS)
    counts = {label: {field: {"absence_codes": 0, "actual_nan": 0} for field in fields}
              for label in masks}
    offset = 0
    for chunk in pd.read_csv(source, usecols=list(fields), chunksize=50000):
        for field in fields:
            absence = ((chunk[field] == -1) if field in BAF_MINUS_ONE_FIELDS
                       else (chunk[field] < 0)).to_numpy()
            actual_nan = chunk[field].isna().to_numpy()
            for label, mask in masks.items():
                local = mask[offset:offset + len(chunk)]
                counts[label][field]["absence_codes"] += int(np.sum(absence & local))
                counts[label][field]["actual_nan"] += int(np.sum(actual_nan & local))
        offset += len(chunk)
    if offset != rows:
        raise ValueError("The CSV population does not match the recorded partitions.")
    for label, mask in masks.items():
        for field in fields:
            counts[label][field]["fraction"] = counts[label][field]["absence_codes"] / int(mask.sum())
    report = {"status": "complete", "raw_file": str(source),
              "raw_file_sha256": provenance["raw_file_sha256"],
              "partition_manifest_sha256": file_sha256(manifest_path),
              "population_rows": {label: int(mask.sum()) for label, mask in masks.items()},
              "absence_counts": counts,
              "primary_policy": "Retain numerical absence codes; actual NaN imputation does not replace these codes.",
              "sensitivity_policy": "Convert only documented codes to NaN, then fit means and indicators within the training partition.",
              "definition_source": "https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf"}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing absence-code audit; choose a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.partitions_dir, args.output), indent=2))
