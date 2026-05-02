import argparse
import cv2
import mediapipe as mp
import sys
from pathlib import Path
from tqdm import tqdm


def crop_faces(
    input_dir: str,
    output_dir: str,
    target_size: int = 224,
    padding: float = 0.2,
) -> None:
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

    processed = skipped_existing = skipped_no_face = errors = 0

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        for img_path in tqdm(image_files, desc="Cropping faces"):
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                skipped_existing += 1
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  ERROR: Could not read {img_path}")
                    errors += 1
                    continue

                h, w = img.shape[:2]
                results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if not results.detections:
                    skipped_no_face += 1
                    continue

                # Use the largest detected face
                best = max(
                    results.detections,
                    key=lambda d: (
                        d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height
                    ),
                )
                bbox = best.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                pad_x = int(bw * padding)
                pad_y = int(bh * padding)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)

                face_crop = img[y1:y2, x1:x2]
                face_resized = cv2.resize(face_crop, (target_size, target_size))
                cv2.imwrite(str(out_path), face_resized)
                processed += 1

            except Exception as e:
                print(f"  ERROR processing {img_path}: {e}")
                errors += 1

    print(
        f"\nDone — processed: {processed}  |  skipped (existing): {skipped_existing}"
        f"  |  no face: {skipped_no_face}  |  errors: {errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop face regions from extracted frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing extracted frames")
    parser.add_argument(
        "--output_dir",
        default="data/processed/faces",
        help="Output directory for face crops (default: data/processed/faces)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Output face image size in pixels (default: 224)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Fractional padding added around the bounding box (default: 0.2)",
    )
    args = parser.parse_args()
    crop_faces(args.input_dir, args.output_dir, args.target_size, args.padding)


if __name__ == "__main__":
    main()
