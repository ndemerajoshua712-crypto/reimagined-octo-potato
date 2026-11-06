import numpy as np
import pandas as pd
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from pathlib import Path


REQUIRED_COLUMNS = {"score", "label", "split"}
CALIBRATION_PATH = Path(__file__).resolve().parent / "cases" / "score_calibration.json"


def evaluate_calibration_csv(source):
    data = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Calibration CSV must contain score, label, and split columns; "
            f"missing: {', '.join(sorted(missing))}."
        )

    data = data.loc[:, ["score", "label", "split"]].dropna().copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna()
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
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (calibrated >= lower) & (
            calibrated <= upper if upper == 1.0 else calibrated < upper
        )
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - calibrated[mask].mean())

    result = {
        "train_rows": int(len(training)),
        "test_rows": int(len(testing)),
        "raw_accuracy": float(accuracy_score(labels, raw_scores >= 0.5)),
        "calibrated_accuracy": float(accuracy_score(labels, calibrated >= 0.5)),
        "raw_brier": float(brier_score_loss(labels, raw_scores)),
        "calibrated_brier": float(brier_score_loss(labels, calibrated)),
        "calibrated_ece": float(ece),
        "calibrated_auc": float(roc_auc_score(labels, calibrated)),
        "heldout_calibrated_scores": calibrated,
        "heldout_labels": labels,
        "calibrator": calibrator,
    }
    return result


def save_calibration(result, path=CALIBRATION_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
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
