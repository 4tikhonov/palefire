import cv2
import sys
import argparse
from pathlib import Path

def split_image(image_path, output_dir="report"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    h, w, _ = img.shape
    mid = w // 2

    # Crop left and right halves
    left = img[:, :mid]
    right = img[:, mid:]

    base_name = Path(image_path).stem
    left_path = Path(output_dir) / f"{base_name}_left.png"
    right_path = Path(output_dir) / f"{base_name}_right.png"

    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)

    print(f"Split {image_path} into:")
    print(f"  - {left_path} ({left.shape[1]}x{left.shape[0]})")
    print(f"  - {right_path} ({right.shape[1]}x{right.shape[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image to split")
    args = parser.parse_args()
    split_image(args.image)
