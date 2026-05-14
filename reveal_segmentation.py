import os

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("./examples_PET")
OUTPUT_DIR = Path("./trimap/PET")

IMAGE_NAMES = [
    "Abyssinian",
    "american_bulldog",
    "basset_hound",
    "beagle",
    "miniature_pinscher",
]

def make_trimap(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, animal = cv2.threshold(
        gray_norm,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((7, 7), np.uint8)
    expand_kernel = np.ones((14, 14), np.uint8)
    expanded_animal = cv2.dilate(animal, expand_kernel, iterations=1)

    dilated = cv2.dilate(expanded_animal, kernel, iterations=1)
    eroded = cv2.erode(expanded_animal, kernel, iterations=1)
    border = cv2.subtract(dilated, eroded)

    trimap = np.zeros_like(expanded_animal, dtype=np.uint8)
    trimap[expanded_animal == 255] = 255
    trimap[border == 255] = 128

    return trimap


def save_colored_trimap(trimap: np.ndarray, output_path: Path) -> None:
    color = np.zeros((trimap.shape[0], trimap.shape[1], 3), dtype=np.uint8)

    color[trimap == 0] = [0, 0, 0]          # background
    color[trimap == 128] = [255, 255, 0]    # border
    color[trimap == 255] = [255, 255, 255]  # animal

    cv2.imwrite(str(output_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))


def save_overlay(base_image_path: Path, trimap: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    base = Image.open(base_image_path).convert("RGB")
    base = np.array(base)

    if base.shape[:2] != trimap.shape[:2]:
        trimap = cv2.resize(
            trimap,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = base.copy()

    # RGB overlay colors
    overlay[trimap == 128] = [255, 255, 0]  # yellow border
    overlay[trimap == 255] = [255, 0, 0]    # red animal

    result = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    for image_name in IMAGE_NAMES:
        input_path = IMAGE_DIR / f"{image_name}_trimap.png"
        base_image_path = IMAGE_DIR / f"{image_name}.jpg"

        trimap = make_trimap(input_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # raw_path = OUTPUT_DIR / f"{image_name}_trimap_raw.png"
        color_path = OUTPUT_DIR / f"{image_name}_trimap_color.png"
        overlay_path = OUTPUT_DIR / f"{image_name}_trimap_overlay.png"

        # cv2.imwrite(str(raw_path), trimap)
        save_colored_trimap(trimap, color_path)
        save_overlay(base_image_path, trimap, overlay_path)

        # print(f"Saved: {raw_path}")
        print(f"Saved: {color_path}")
        print(f"Saved: {overlay_path}")