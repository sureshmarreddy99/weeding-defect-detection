import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


def process_image(input_path, output_path):
    """
    Temporary starter function.
    Reads an image, writes a basic output image.
    Later we will replace this body with your notebook logic.
    """
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    output_img = img_bgr.copy()

    cv2.putText(
        output_img,
        "Processing pipeline connected",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)

    return {
        "output_path": output_path,
        "defect_count": 0,
        "defects": []
    }
