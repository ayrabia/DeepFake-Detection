import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm


def compute_fft(input_dir: str, output_dir: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    image_files = (
        list(input_dir.rglob("*.jpg"))
        + list(input_dir.rglob("*.jpeg"))
        + list(input_dir.rglob("*.png"))
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image file(s)")

    processed = skipped = errors = 0

    for img_path in tqdm(image_files, desc="Computing FFT"):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ERROR: Could not read {img_path}")
                errors += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Center-shifted log-magnitude spectrum
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = np.log1p(np.abs(f_shift))

            # Normalize to [0, 255] and save as JPEG with the same filename
            magnitude_uint8 = cv2.normalize(
                magnitude, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imwrite(str(out_path), magnitude_uint8)
            processed += 1

        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            errors += 1

    print(
        f"\nDone — processed: {processed}  |  skipped (existing): {skipped}  |  errors: {errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FFT magnitude spectrum images from face crops."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing face crops")
    parser.add_argument(
        "--output_dir",
        default="data/processed/fft",
        help="Output directory for FFT images (default: data/processed/fft)",
    )
    args = parser.parse_args()
    compute_fft(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
