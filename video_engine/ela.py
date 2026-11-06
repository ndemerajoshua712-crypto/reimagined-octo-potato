import cv2
import numpy as np


def error_level_analysis(image_path, quality=90):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image for ELA: {image_path}")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise OSError("Could not encode image for ELA.")

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    difference = cv2.absdiff(image, recompressed)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    p99 = float(np.percentile(gray_difference, 99))
    scale = 255.0 / max(p99, 1.0)
    visualization = cv2.convertScaleAbs(difference, alpha=scale)

    return {
        "visualization": cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),
        "mean_difference": float(np.mean(gray_difference)),
        "p95_difference": float(np.percentile(gray_difference, 95)),
        "quality": int(quality),
    }
