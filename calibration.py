import numpy as np
import pandas as pd
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from pathlib import Path


REQUIRED_COLUMNS = {"score", "label", "split"}
GROUP_COLUMNS = ("dataset", "manipulation_type")
CALIBRATION_PATH = Path(__file__).resolve().parent / "cases" / "score_calibration.json"


def _expected_calibration_error(labels, scores, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (scores >= lower) & (scores <= upper if upper == 1.0 else scores < upper)
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return float(error)


def _metrics(labels, raw_scores, calibrated_scores):
    metrics = {
        "rows": int(len(labels)),
        "raw_brier": float(brier_score_loss(labels, raw_scores)),
        "calibrated_brier": float(brier_score_loss(labels, calibrated_scores)),
        "calibrated_ece": _expected_calibration_error(labels, calibrated_scores),
    }
    if np.unique(labels).size == 2:
        metrics["raw_auc"] = float(roc_auc_score(labels, raw_scores))
        metrics["calibrated_auc"] = float(roc_auc_score(labels, calibrated_scores))
    return metrics


def evaluate_calibration_csv(source):
    data = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Calibration CSV must contain score, label, and split columns; "
            f"missing: {', '.join(sorted(missing))}."
        )

    kept_columns = ["score", "label", "split", *[c for c in GROUP_COLUMNS if c in data.columns]]
    data = data.loc[:, kept_columns].dropna(subset=["score", "label", "split"]).copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    # Optional subgroup labels may be blank; retain those rows for overall metrics.
    data = data.dropna(subset=["score", "label"])
    data["split"] = data["split"].astype(str).str.strip().str.lower()

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 (real) or 1 (manipulated).")
    if not data["score"].between(0, 1).all():
        raise ValueError("Scores must be probabilities between 0 and 1.")

    training = data[data["split"] == "train"]
    testing = data[data["split"].isin(["test", "holdout", "validation"])]
    if len(training) < 10 or training["label"].nunique() != 2:
        raise ValueError("Training split needs at least 10 rows and both labels.")
    if len(testing) < 10 or testing["label"].nunique() != 2:
        raise ValueError("Held-out split needs at least 10 rows and both labels.")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(training["score"].to_numpy(), training["label"].to_numpy())
    raw_scores = testing["score"].to_numpy(dtype=float)
    labels = testing["label"].to_numpy(dtype=int)
    calibrated = calibrator.predict(raw_scores)
    metrics = _metrics(labels, raw_scores, calibrated)
    subgroup_metrics = {}
    for column in GROUP_COLUMNS:
        if column not in testing:
            continue
        values = testing[column].fillna("unspecified").astype(str).str.strip()
        values = values.replace("", "unspecified")
        grouped = {}
        for value in sorted(values.unique()):
            mask = values.to_numpy() == value
            group_labels = labels[mask]
            if len(group_labels) < 10:
                continue
            grouped[value] = _metrics(group_labels, raw_scores[mask], calibrated[mask])
        subgroup_metrics[column] = grouped
    result = {
        "train_rows": int(len(training)),
        "test_rows": int(len(testing)),
        "raw_accuracy": float(accuracy_score(labels, raw_scores >= 0.5)),
        "calibrated_accuracy": float(accuracy_score(labels, calibrated >= 0.5)),
        **metrics,
        "subgroup_metrics": subgroup_metrics,
        "heldout_calibrated_scores": calibrated,
        "heldout_labels": labels,
        "calibrator": calibrator,
    }
    return result


def save_calibration(result, path=CALIBRATION_PATH, source_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "method": "isotonic_regression",
        "source_sha256": source_sha256,
        "x_thresholds": result["calibrator"].X_thresholds_.tolist(),
        "y_thresholds": result["calibrator"].y_thresholds_.tolist(),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "raw_accuracy": result["raw_accuracy"],
        "calibrated_accuracy": result["calibrated_accuracy"],
        "raw_brier": result["raw_brier"],
        "calibrated_brier": result["calibrated_brier"],
        "calibrated_ece": result["calibrated_ece"],
        "calibrated_auc": result["calibrated_auc"],
        "subgroup_metrics": result["subgroup_metrics"],
    }
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def load_calibration(path=CALIBRATION_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(score, calibration):
    if calibration is None:
        return None
    return float(
        np.interp(
            float(score),
            calibration["x_thresholds"],
            calibration["y_thresholds"],
        )
    )
