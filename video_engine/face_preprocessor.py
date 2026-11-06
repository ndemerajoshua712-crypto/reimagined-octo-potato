import cv2
import numpy as np


def preprocess_face(image_path, image_size=(224, 224)):
    """
    Prepare a detected face for an AI model.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read face image: {image_path}")

    # OpenCV loads images as BGR.
    # Convert to RGB for most AI models.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the model's expected dimensions.
    image = cv2.resize(image, image_size)

    # Convert pixel values from 0-255 to 0-1.
    image = image.astype(np.float32) / 255.0

    return image


def prepare_for_model(image_path, image_size=(224, 224)):
    """
    Prepare a face and add the batch dimension.
    """

    image = preprocess_face(
        image_path,
        image_size
    )

    # Shape:
    # (224, 224, 3)
    #
    # becomes:
    # (1, 224, 224, 3)

    image = np.expand_dims(image, axis=0)

    return image