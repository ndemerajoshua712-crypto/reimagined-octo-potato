def calculate_evidence_score(results):
    """
    Calculate an overall fake-evidence score
    from all detected faces.

    The score is based on the model's actual
    fake probabilities, not just REAL/FAKE labels.
    """

    fake_probabilities = []

    for frame in results:

        for face in frame["faces"]:

            fake_probability = face.get(
                "fake_probability"
            )

            if fake_probability is not None:
                fake_probabilities.append(
                    fake_probability
                )

    # No faces detected
    if not fake_probabilities:
        return {
            "status": "no_faces",
            "evidence_score": None,
            "faces_analyzed": 0,
            "risk_level": "Unknown"
        }

    # Average fake probability
    average_fake_probability = (
        sum(fake_probabilities)
        / len(fake_probabilities)
    )

    evidence_score = average_fake_probability * 100

    # Determine risk level
    if evidence_score >= 75:
        risk_level = "High"

    elif evidence_score >= 55:
        risk_level = "Moderate"

    elif evidence_score >= 45:
        risk_level = "Low"

    else:
        risk_level = "Very Low"

    return {
        "status": "success",
        "evidence_score": evidence_score,
        "faces_analyzed": len(fake_probabilities),
        "risk_level": risk_level
    }