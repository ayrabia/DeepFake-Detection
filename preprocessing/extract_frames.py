import argparse
import cv2
import sys
from pathlib import Path
from tqdm import tqdm


def extract_frames(input_dir: str, output_dir: str, frame_interval: int = 5) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    video_extensions = {".mp4", ".avi", ".flv"}
    video_files = [
        p for ext in video_extensions for p in input_dir.rglob(f"*{ext}")
    ]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} video file(s) — sampling every {frame_interval}th frame")

    processed = skipped = errors = 0

    for video_path in tqdm(video_files, desc="Extracting frames"):
        rel_path = video_path.relative_to(input_dir)
        video_name = video_path.stem
        out_subdir = output_dir / rel_path.parent / video_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Skip if any frames already exist for this video
        if any(out_subdir.glob(f"{video_name}_frame_*.jpg")):
            skipped += 1
            continue

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  ERROR: Could not open {video_path}")
                errors += 1
                continue

            frame_count = saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    out_path = out_subdir / f"{video_name}_frame_{frame_count:05d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_count += 1
                frame_count += 1

            cap.release()
            processed += 1

        except Exception as e:
            print(f"  ERROR processing {video_path}: {e}")
            errors += 1

    print(
        f"\nDone — processed: {processed}  |  skipped (already done): {skipped}  |  errors: {errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract JPEG frames from video files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw video files")
    parser.add_argument(
        "--output_dir",
        default="data/processed/frames",
        help="Output directory for extracted frames (default: data/processed/frames)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Save every Nth frame (default: 5, i.e. 6 fps from 30 fps source)",
    )
    args = parser.parse_args()
    extract_frames(args.input_dir, args.output_dir, args.frame_interval)


if __name__ == "__main__":
    main()
