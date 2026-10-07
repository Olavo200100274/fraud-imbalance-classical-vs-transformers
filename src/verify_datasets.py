"""Verify raw study inputs without importing ML libraries or modifying files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(directory, manifest, names=None):
    records = {record["name"]: record for record in manifest["files"]}
    selected = list(records) if names is None else names
    unknown = set(selected) - records.keys()
    if unknown:
        raise ValueError(f"Unknown dataset filenames: {sorted(unknown)}")
    failures = []
    for name in selected:
        path = directory / name
        if not path.is_file():
            failures.append(f"{name}: missing; obtain the CSV from the official source")
            continue
        record = records[name]
        if path.stat().st_size != record["bytes"]:
            failures.append(f"{name}: unexpected byte size")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            failures.append(f"{name}: SHA-256 mismatch")
        else:
            print(f"OK {name}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "datasets")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifest.json")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        failures = verify(args.directory, manifest, args.files)
    except ValueError as error:
        parser.error(str(error))
    for failure in failures:
        print(f"FAIL {failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
