"""
image_processor.py
--------------------
MODALITY 4: Image (applicant ID / document photo).

A lightweight computer-vision quality & authenticity pre-check for the
uploaded ID/document image: resolution check, blur (focus) detection via
the variance-of-Laplacian method, and basic brightness check. This acts as
an automated "is this document usable/legible" gate before a human reviews
the application -- a common real-world co-pilot pattern (reject blurry or
unreadable uploads automatically, flag borderline ones for human review).

Implemented with Pillow + scikit-image only (no extra installs needed).
"""

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import laplace
import io


def analyze_document_image(file_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = img.size
    arr = np.asarray(img) / 255.0
    gray = rgb2gray(arr)

    # Sharpness: variance of the Laplacian. Higher = sharper.
    lap = laplace(gray)
    sharpness = float(lap.var())

    brightness = float(gray.mean())

    low_res = width < 400 or height < 300
    blurry = sharpness < 0.0008
    too_dark = brightness < 0.20
    too_bright = brightness > 0.92

    issues = []
    if low_res:
        issues.append("resolution too low")
    if blurry:
        issues.append("image appears blurry / out of focus")
    if too_dark:
        issues.append("image too dark")
    if too_bright:
        issues.append("image overexposed")

    if not issues:
        quality = "Good"
    elif len(issues) == 1:
        quality = "Borderline"
    else:
        quality = "Poor"

    return {
        "width": width,
        "height": height,
        "sharpness_score": round(sharpness, 6),
        "brightness_score": round(brightness, 3),
        "quality": quality,
        "issues": issues,
        "usable": quality != "Poor",
    }
