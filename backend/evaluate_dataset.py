"""Run the analysis pipeline over a labeled manifest for offline evaluation.

Example:
    python -m backend.evaluate_dataset labeled_manifest.csv --output predictions.csv

Manifest columns: video_path,label,split; dataset and manipulation_type are
optional. Relative video paths are resolved from the manifest's folder.
"""

import argparse
import csv
import tempfile
from pathlib import Path


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"video_path", "label", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest must contain video_path, label, and split columns; "
                f"missing: {', '.join(sorted(missing))}."
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Manifest row {row_number} has an invalid label.") from error
            if label not in (0, 1):
                raise ValueError(f"Manifest row {row_number} label must be 0 or 1.")
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "test", "holdout", "validation"}:
                raise ValueError(
                    f"Manifest row {row_number} split must be train, test, holdout, or validation."
                )
            supplied_path = Path((row.get("video_path") or "").strip())
            video_path = supplied_path if supplied_path.is_absolute() else Path(path).resolve().parent / supplied_path
            if not video_path.is_file():
                raise ValueError(f"Manifest row {row_number} video file was not found.")
            rows.append({
                "video_path": video_path.resolve(),
                "label": label,
                "split": split,
                "dataset": (row.get("dataset") or "unspecified").strip() or "unspecified",
                "manipulation_type": (row.get("manipulation_type") or "unspecified").strip() or "unspecified",
            })
    if not rows:
        raise ValueError("Manifest contains no video rows.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV manifest of labeled media")
    parser.add_argument("--output", type=Path, default=Path("cases/labeled_predictions.csv"))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")

    from analysis_pipeline import analyze_video_file
    from video_engine.provenance import sha256_file

    rows = read_manifest(args.manifest)
    seen_hash_splits = {}
    predictions = []
    paired_count = 0

    for index, item in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] Analyzing {item['video_path'].name}", flush=True)
        media_hash = sha256_file(item["video_path"])
        prior_split = seen_hash_splits.get(media_hash)
        if prior_split is not None:
            if prior_split != item["split"]:
                raise ValueError(
                    "Identical video files appear in multiple splits; keep duplicates in one split."
                )
            raise ValueError(
                "The same video appears more than once in the manifest; remove duplicate rows."
            )
        seen_hash_splits[media_hash] = item["split"]
        try:
            with tempfile.TemporaryDirectory(prefix="deeptrace-eval-") as output_dir:
                result = analyze_video_file(
                    item["video_path"], item["video_path"].name,
                    output_root=Path(output_dir),
                    source_sha256=media_hash,
                )
            modalities = result["multimodal_result"].get("available_modalities", [])
            # The global calibrator is fitted only on paired audio+video scores.
            combined = result["multimodal_result"].get("primary_score") if len(modalities) == 2 else None
            if combined is not None:
                paired_count += 1
            predictions.append({
                "score": "" if combined is None else f"{combined:.8f}",
                "label": item["label"],
                "split": item["split"],
                "dataset": item["dataset"],
                "manipulation_type": item["manipulation_type"],
                "video_sha256": media_hash,
                "filename": item["video_path"].name,
                "video_score": "" if result["multimodal_result"].get("video_evidence") is None else f"{result['multimodal_result']['video_evidence']:.8f}",
                "audio_score": "" if result["multimodal_result"].get("audio_evidence") is None else f"{result['multimodal_result']['audio_evidence']:.8f}",
                "modalities": "+".join(modalities),
                "analysis_error": "",
            })
        except Exception as error:
            predictions.append({
                "score": "", "label": item["label"], "split": item["split"],
                "dataset": item["dataset"], "manipulation_type": item["manipulation_type"],
                "video_sha256": media_hash, "filename": item["video_path"].name,
                "video_score": "", "audio_score": "", "modalities": "",
                "analysis_error": type(error).__name__,
            })
            print(f"  Analysis failed ({type(error).__name__}).", flush=True)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "score", "label", "split", "dataset", "manipulation_type",
        "video_sha256", "filename", "video_score", "audio_score",
        "modalities", "analysis_error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} rows ({paired_count} paired model scores) to {output_path}")
    print("Use only paired scores for the global calibrator; inspect analysis_error and modality columns first.")


if __name__ == "__main__":
    main()
