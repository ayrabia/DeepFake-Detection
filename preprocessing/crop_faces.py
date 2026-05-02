import argparse
import cv2
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MODEL_PATH = Path(__file__).parent / "blaze_face_short_range.tflite"


def _ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"Downloading face detection model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return str(MODEL_PATH)


def crop_faces(
    input_dir: str,
    output_dir: str,
    target_size: int = 224,
    padding: float = 0.2,
) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

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

    model_path = _ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    processed = skipped_existing = skipped_no_face = errors = 0

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
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.detections:
                skipped_no_face += 1
                continue

            # Use the largest detected face
            best = max(
                result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            bbox = best.bounding_box
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

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
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/processed/faces")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.2)
    args = parser.parse_args()
    crop_faces(args.input_dir, args.output_dir, args.target_size, args.padding)


if __name__ == "__main__":
    main()
