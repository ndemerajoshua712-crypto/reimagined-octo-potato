from datetime import datetime


def generate_forensic_report(
    filename,
    multimodal_result,
    video_result,
    audio_result,
    temporal_result,
    sync_result,
    calibrated_score=None,
    calibration_info=None,
):

    report = []

    report.append("=" * 70)
    report.append("DEEPTRACE FORENSIC ANALYSIS REPORT")
    report.append("=" * 70)

    report.append(f"File: {filename}")
    report.append(
        f"Analysis Time: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    report.append("\n" + "-" * 70)
    report.append("FINAL ASSESSMENT")
    report.append("-" * 70)

    report.append(
        f"Verdict: {multimodal_result['verdict']}"
    )

    report.append(
        f"Evidence Score: "
        f"{multimodal_result['primary_score']*100:.2f}%"
    )

    if calibrated_score is not None:
        report.append(f"Calibrated Estimate: {calibrated_score*100:.2f}%")
        if calibration_info:
            report.append(
                "Calibration: "
                f"{calibration_info.get('method', 'isotonic_regression')}; "
                f"trained on {calibration_info.get('train_rows', 'unknown')} rows; "
                f"evaluated on {calibration_info.get('test_rows', 'unknown')} held-out rows."
            )
        report.append(
            "NOTE: This calibrated estimate is valid only for data similar to the "
            "calibration set. It does not establish authenticity or manipulation."
        )
        subgroup_metrics = (calibration_info or {}).get("subgroup_metrics", {})
        rendered_groups = 0
        for dimension, groups in subgroup_metrics.items():
            for group, metrics in groups.items():
                auc = metrics.get("calibrated_auc")
                suffix = f"; AUC {auc:.3f}" if auc is not None else ""
                report.append(
                    f"Held-out {dimension}={group}: {metrics['rows']} rows; "
                    f"Brier {metrics['calibrated_brier']:.3f}; "
                    f"ECE {metrics['calibrated_ece']:.3f}{suffix}."
                )
                rendered_groups += 1
        if rendered_groups:
            report.append(
                "Subgroup results are descriptive; small groups may be uncertain."
            )

    report.append(
        "\nNOTE: The evidence score is a prototype "
        "multimodal score and should not be interpreted "
        "as a calibrated probability."
    )
    report.append(
        f"Temporal Status: "

        f"{multimodal_result['temporal_status']}"
    )

    report.append(f"A/V sync status: "

                  f"{multimodal_result['av_sync_status']}")

    report.append("\n" + "-" * 70)
    report.append("VIDEO ANALYSIS")
    report.append("-" * 70)

    report.append(
        f"Mean Fake Score: "
        f"{video_result['mean_fake_probability'] * 100:.2f}%"
    )

    report.append(
        f"Frames Analyzed: "
        f"{video_result['frames_analyzed']}"
    )

    report.append(
        f"Faces Analyzed: "
        f"{video_result['faces_analyzed']}"
    )

    report.append("\n" + "-" * 70)
    report.append("AUDIO ANALYSIS")
    report.append("-" * 70)

    report.append(
        f"Average Spoof Score: "
        f"{audio_result['average_spoof_probability'] * 100:.2f}%"
    )

    report.append(
        f"Maximum Spoof Score: "
        f"{audio_result['maximum_spoof_probability'] * 100:.2f}%"
    )

    report.append(
        f"Suspicious Windows: "
        f"{audio_result['spoof_window_percentage']:.2f}%"
    )

    report.append("\n" + "-" * 70)
    report.append("TEMPORAL ANALYSIS")
    report.append("-" * 70)

    report.append(
        f"Temporal Consistency: "
        f"{temporal_result['temporal_consistency'] * 100:.2f}%"
    )

    report.append(
        f"Average Deviation: "
        f"{temporal_result['average_deviation'] * 100:.2f}%"
    )

    report.append("\n" + "-" * 70)
    report.append("A/V SYNCHRONIZATION")
    report.append("-" * 70)

    report.append(
        f"Video Duration: "
        f"{sync_result['video_duration']:.2f} seconds"
    )

    report.append(
        f"Audio Duration: "
        f"{sync_result['audio_duration']:.2f} seconds"
    )

    report.append(
        f"Duration Difference: "
        f"{sync_result['duration_difference']:.2f} seconds"
    )

    report.append(
        f"Sync Status: "
        f"{sync_result['sync_status']}"
    )

    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    return "\n".join(report)
