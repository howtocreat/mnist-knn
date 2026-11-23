import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from dataset import FEATURE_COUNT, FULL_DATASET, load_split
from knn import CustomKNNClassifier


def read_grayscale(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        if np.all(alpha == 255):
            return np.asarray(image.convert("L"))
        visible = alpha > 0
        if not visible.any():
            raise ValueError("The image is fully transparent.")
        grayscale = np.asarray(rgba.convert("L"))
        background = 255 if np.median(grayscale[visible]) < 128 else 0
        canvas = Image.new("RGBA", rgba.size, (background,) * 3 + (255,))
        return np.asarray(Image.alpha_composite(canvas, rgba).convert("L"))


def prepare_digit(image):
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    background = float(np.median(border))
    light_foreground = float(image.max()) - background
    dark_foreground = background - float(image.min())
    if max(light_foreground, dark_foreground) < 24:
        raise ValueError("No clear digit found. Use a high-contrast image.")

    pixels = image.astype(np.float32)
    contrast = pixels - background if light_foreground > dark_foreground else background - pixels
    contrast = np.clip(contrast * (255 / max(light_foreground, dark_foreground)), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if count == 1 or areas.max() < 4:
        raise ValueError("No clear digit found. Use a tightly cropped image.")
    keep = np.r_[False, areas >= max(2, areas.max() * 0.02)]
    foreground = keep[labels]
    rows, columns = np.nonzero(foreground)
    height = rows.max() - rows.min() + 1
    width = columns.max() - columns.min() + 1
    contrast[~foreground] = 0
    crop = contrast[rows.min():rows.max() + 1, columns.min():columns.max() + 1]

    scale = 20 / max(height, width)
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, dimensions, interpolation=method)
    moments = cv2.moments(resized)
    if not moments["m00"]:
        raise ValueError("The digit is too thin to resize.")
    x = round(13.5 - moments["m10"] / moments["m00"])
    y = round(13.5 - moments["m01"] / moments["m00"])
    x = max(0, min(x, 28 - dimensions[0]))
    y = max(0, min(y, 28 - dimensions[1]))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[y:y + dimensions[1], x:x + dimensions[0]] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Recognize one handwritten digit in an image.")
    parser.add_argument("image", type=Path, help="Path to a PNG, JPEG, or another Pillow-readable image")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors (default: 3)")
    args = parser.parse_args()
    try:
        image = prepare_digit(read_grayscale(args.image))
        training, labels = load_split(FULL_DATASET, "train")
        classifier = CustomKNNClassifier(k=args.k).fit(training, labels)
        digit = classifier.predict(image.reshape(1, FEATURE_COUNT).astype(np.float32) / 255)[0]
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(2, f"Could not predict a digit: {exc}\n")
    print(f"Predicted digit: {digit}")


if __name__ == "__main__":
    main()
