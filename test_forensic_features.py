import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from calibration import (
    apply_calibration,
    evaluate_calibration_csv,
    load_calibration,
    save_calibration,
)
from case_history import list_cases, save_case
from video_engine.ela import error_level_analysis
from audio_engine.lipsync_analyzer import compute_sync, lip_sync_unavailable


def _calibration_csv():
    rows = ["score,label,split"]
    for split in ("train", "test"):
        for index in range(20):
            label = index % 2
            score = 0.1 + 0.8 * label + (index % 4) * 0.01
            rows.append(f"{score},{label},{split}")
    return io.StringIO("\n".join(rows))


class ForensicFeatureTests(unittest.TestCase):
    def test_calibration_reports_heldout_metrics(self):
        metrics = evaluate_calibration_csv(_calibration_csv())

        self.assertEqual(metrics["train_rows"], 20)
        self.assertEqual(metrics["test_rows"], 20)
        self.assertGreaterEqual(metrics["calibrated_accuracy"], 0)
        self.assertLessEqual(metrics["calibrated_accuracy"], 1)
        self.assertGreaterEqual(metrics["calibrated_brier"], 0)
        self.assertLessEqual(metrics["calibrated_brier"], 1)
        self.assertGreaterEqual(metrics["calibrated_ece"], 0)
        self.assertLessEqual(metrics["calibrated_ece"], 1)

    def test_calibration_mapping_round_trip(self):
        metrics = evaluate_calibration_csv(_calibration_csv())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            saved = save_calibration(metrics, path)
            loaded = load_calibration(path)

            self.assertEqual(saved, loaded)
            self.assertGreaterEqual(apply_calibration(0.5, loaded), 0)
            self.assertLessEqual(apply_calibration(0.5, loaded), 1)

    def test_calibration_rejects_missing_holdout_split(self):
        rows = ["score,label,split"]
        rows.extend(
            f"{0.2 + label * 0.6},{label},train"
            for label in (0, 1)
            for _ in range(6)
        )
        csv = io.StringIO("\n".join(rows))

        with self.assertRaisesRegex(ValueError, "Held-out split"):
            evaluate_calibration_csv(csv)

    def test_case_history_persists_hash_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cases.sqlite3"
            result = {
                "multimodal_result": {
                    "verdict": "LOW",
                    "primary_score": 0.25,
                },
                "video_results": [{"face_count": 2}],
                "audio_windows": {"total_windows": 3},
            }

            saved = save_case("sample.mp4", "a" * 64, result, database_path=database)
            cases = list_cases(database_path=database)

            self.assertEqual(cases[0]["case_id"], saved["case_id"])
            self.assertEqual(cases[0]["sha256"], "a" * 64)
            self.assertEqual(cases[0]["face_count"], 2)
            self.assertEqual(cases[0]["audio_windows"], 3)

    def test_ela_returns_rgb_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.jpg"
            image = np.zeros((64, 80, 3), dtype=np.uint8)
            image[16:48, 20:60] = (20, 100, 220)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            result = error_level_analysis(image_path)

            self.assertEqual(result["visualization"].shape, (64, 80, 3))
            self.assertEqual(result["quality"], 90)
            self.assertGreaterEqual(result["mean_difference"], 0)
            self.assertGreaterEqual(result["p95_difference"], result["mean_difference"])

    def test_lip_sync_correlates_aligned_signals(self):
        time_axis = np.arange(500, dtype=np.float64) / 25
        mouth = np.sin(2 * np.pi * 1.2 * time_axis)
        audio = np.sin(2 * np.pi * 1.2 * time_axis)

        result = compute_sync(mouth, audio, 25)

        self.assertGreater(result["peak_correlation"], 0.7)
        self.assertEqual(result["audio_delay_ms"], 0.0)
        self.assertEqual(len(result["mouth_trace"]), len(mouth))

    def test_lip_sync_abstains_when_no_face_is_tracked(self):
        mouth = np.full(150, np.nan)
        audio = np.ones(150)

        result = compute_sync(mouth, audio, 25)
        fallback = lip_sync_unavailable("model unavailable")

        self.assertEqual(result["status"], "INSUFFICIENT DATA")
        self.assertEqual(result["face_coverage"], 0.0)
        self.assertEqual(len(result["mouth_trace"]), len(mouth))
        self.assertEqual(fallback["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
