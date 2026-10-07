# Evidence release

This release accompanies the final October 2026 thesis. It contains the complete 72-cell primary metric grid, frozen configurations, validation summaries, 12 threshold studies, corrected BAF transfer metrics, predefined sensitivity controls, aggregate interpretability evidence, and thesis tables/figures.

The original manifest and reporting QA are immutable author-side records; their SHA-256 identities are pinned in release_manifest.json. The original manifest's administrative completion flags were not rewritten. The release inventory instead verifies the selected completed evidence. Machine-specific path prefixes are normalised; both the original and public file hashes are recorded. Archive-relative paths refer to author-retained material, not files promised in this public release. Historical BAF final-model selection is explicit.

Per-row TEST/validation predictions, model weights, raw attribution matrices, local logs, checkpoints and editorial records are not distributed here. Hashes identify that evidence but are not a substitute for its contents. Accordingly this bundle supports inspection and verification of reported metrics, tables, plots and provenance, not independent exact recomputation of historical fitted models from this bundle alone. Compact PR coordinates are for plotting only; reported average precision is recovered from complete frozen scores.

Verify public file integrity locally with `python src/export_public_evidence.py --verify`. This is not a cross-computer reproducibility claim. The author-side export requires the retained local result archives.
