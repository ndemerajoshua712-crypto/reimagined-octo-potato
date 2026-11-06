import cv2
import os


CASCADE_FILES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
)
face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + filename)
    for filename in CASCADE_FILES
]


def _detect_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    candidates = []

    for cascade in face_cascades:
        candidates.extend(
            tuple(map(int, box))
            for box in cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
            )
        )

    profile = face_cascades[2].detectMultiScale(
        cv2.flip(gray, 1),
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
    )
    image_width = image.shape[1]
    candidates.extend(
        (image_width - x - width, y, width, height)
        for x, y, width, height in profile
    )

    candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
    results = []

    for candidate in candidates:
        x, y, width, height = candidate
        duplicate = False

        for kept_x, kept_y, kept_width, kept_height in results:
            left = max(x, kept_x)
            top = max(y, kept_y)
            right = min(x + width, kept_x + kept_width)
            bottom = min(y + height, kept_y + kept_height)
            intersection = max(0, right - left) * max(0, bottom - top)
            smaller_area = min(width * height, kept_width * kept_height)

            if smaller_area and intersection / smaller_area >= 0.65:
                duplicate = True
                break

        if not duplicate:
            results.append(candidate)

    return results


def detect_faces(image_path):
    """
    Detect faces in an image.

    Returns a list containing:
        x, y, width, height
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = _detect_boxes(image)

    results = []

    for x, y, width, height in faces:

        results.append({
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height)
        })

    return results


def extract_faces(
    image_path,
    output_folder,
    frame_name,
    detected_faces=None,
):
    """
    Detect and crop faces from an image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = (
        _detect_boxes(image)
        if detected_faces is None
        else [
            (face["x"], face["y"], face["width"], face["height"])
            for face in detected_faces
        ]
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    extracted = []

    for index, (x, y, width, height) in enumerate(
        faces
    ):

        face = image[
            y:y + height,
            x:x + width
        ]

        filename = os.path.join(
            output_folder,
            f"{frame_name}_face_{index + 1}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        extracted.append({
            "face": index + 1,
            "path": filename,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height)
        })

    return extracted