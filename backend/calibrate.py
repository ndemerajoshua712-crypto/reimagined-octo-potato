"""Fit and save the global score calibrator from labeled prediction rows.

Run from the repository root with the analysis environment active:
    python -m backend.calibrate path\to\labeled_scores.csv

The CSV must contain score,label,split columns. Labels are 0 (real) and
1 (manipulated); split is train for fitting and test/holdout/validation for
held-out evaluation. Do not use this command on production case labels as a
substitute for an independent evaluation set.
"""

import argparse
import hashlib
from pathlib import Path

from calibration import evaluate_calibration_csv, save_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV of labeled raw scores and split assignments")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (defaults to cases/score_calibration.json)",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    source_sha256 = hashlib.sha256(args.csv_path.read_bytes()).hexdigest()
    metrics = evaluate_calibration_csv(args.csv_path)
    if args.output:
        artifact = save_calibration(metrics, args.output, source_sha256=source_sha256)
    else:
        artifact = save_calibration(metrics, source_sha256=source_sha256)

    print(f"Calibration saved: {args.output or 'cases/score_calibration.json'}")
    print(f"Method: {artifact['method']}")
    print(f"Training rows: {artifact['train_rows']}; held-out rows: {artifact['test_rows']}")
    print(
        "Held-out Brier score (raw -> calibrated): "
        f"{artifact['raw_brier']:.4f} -> {artifact['calibrated_brier']:.4f}"
    )
    print(f"Held-out expected calibration error: {artifact['calibrated_ece']:.4f}")
    print(f"Held-out ROC AUC: {artifact['calibrated_auc']:.4f}")
    for dimension, groups in artifact.get("subgroup_metrics", {}).items():
        for name, group in groups.items():
            auc = f"; AUC {group['calibrated_auc']:.4f}" if "calibrated_auc" in group else ""
            print(
                f"{dimension}={name}: {group['rows']} rows; "
                f"Brier {group['raw_brier']:.4f} -> {group['calibrated_brier']:.4f}"
                f"{auc}"
            )
    print("These metrics describe only this labeled dataset and its split.")


if __name__ == "__main__":
    main()
