"""Extract fixed-length frame sequences from cropped face directories.

Reads the existing train/val/test CSVs, groups face crops by source video,
samples exactly 16 evenly-spaced frames per video, and saves each sequence
as a numpy array of shape (16, 224, 224, 3) to data/processed/sequences/.

Also adds a 'sequence_path' column to each CSV so all three paths
(frame_path, fft_path, sequence_path) are available in one place.

Run this after crop_faces.py and before (or after) build_dataset.py.

Pipeline position:
  extract_frames → crop_faces → compute_fft → extract_sequences → build_dataset
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SEQUENCE_LENGTH = 16
FRAME_SIZE = (224, 224)


def sample_indices(n_frames: int, n_samples: int) -> list[int]:
    """Return n_samples evenly-spaced indices into a list of length n_frames."""
    if n_frames <= n_samples:
        # Pad by repeating last frame index
        indices = list(range(n_frames))
        while len(indices) < n_samples:
            indices.append(n_frames - 1)
        return indices
    step = n_frames / n_samples
    return [int(i * step) for i in range(n_samples)]


def build_sequence(frame_paths: list[Path], n_samples: int) -> np.ndarray:
    """Load and stack sampled frames into (n_samples, H, W, 3) uint8 array."""
    sorted_paths = sorted(frame_paths, key=lambda p: p.name)
    indices = sample_indices(len(sorted_paths), n_samples)
    frames = []
    for idx in indices:
        img = cv2.imread(str(sorted_paths[idx]))
        if img is None:
            # Fallback: black frame rather than crash
            img = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != FRAME_SIZE:
                img = cv2.resize(img, FRAME_SIZE)
        frames.append(img)
    return np.stack(frames, axis=0)  # (16, 224, 224, 3)


def process_split(
    csv_path: Path,
    output_dir: Path,
    n_samples: int = SEQUENCE_LENGTH,
) -> None:
    df = pd.read_csv(csv_path)

    if "frame_path" not in df.columns:
        print(f"  Skipping {csv_path.name} — no frame_path column")
        return

    # Already has sequence_path and all are filled → nothing to do
    if "sequence_path" in df.columns and df["sequence_path"].notna().all():
        print(f"  {csv_path.name}: all sequences already present, skipping")
        return

    # Group frames by video (parent directory of each frame_path)
    df["video_dir"] = df["frame_path"].apply(lambda p: str(Path(p).parent))
    video_groups = df.groupby("video_dir")

    seq_map: dict[str, str] = {}  # video_dir → sequence .npy path
    processed = skipped = errors = 0

    for video_dir, group in tqdm(video_groups, desc=f"  {csv_path.name}", leave=False):
        video_dir_path = Path(video_dir)

        # Mirror output directory structure from faces → sequences
        # Find a common base to strip — try faces/ or processed/faces/
        try:
            # Walk up until we find 'faces' in the path components
            parts = video_dir_path.parts
            faces_idx = next(
                (i for i, p in enumerate(parts) if p == "faces"), None
            )
            if faces_idx is not None:
                rel_parts = parts[faces_idx + 1:]
            else:
                rel_parts = (video_dir_path.name,)
            out_dir = output_dir / Path(*rel_parts).parent
        except (StopIteration, TypeError):
            out_dir = output_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        seq_filename = video_dir_path.name + ".npy"
        seq_path = out_dir / seq_filename

        if seq_path.exists():
            skipped += 1
            seq_map[video_dir] = str(seq_path)
            continue

        frame_paths = list(video_dir_path.glob("*.jpg")) + list(
            video_dir_path.glob("*.jpeg")
        )
        if not frame_paths:
            print(f"  WARNING: no frames found in {video_dir}")
            errors += 1
            continue

        try:
            sequence = build_sequence(frame_paths, n_samples)  # (16, 224, 224, 3)
            np.save(str(seq_path), sequence)
            seq_map[video_dir] = str(seq_path)
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {video_dir}: {e}")
            errors += 1

    print(
        f"  Done — processed: {processed}  |  "
        f"skipped (existing): {skipped}  |  errors: {errors}"
    )

    # Update CSV with sequence_path column
    df["sequence_path"] = df["video_dir"].map(seq_map)
    df = df.drop(columns=["video_dir"])

    missing = df["sequence_path"].isna().sum()
    if missing:
        print(f"  Warning: {missing} rows have no sequence (videos with 0 frames)")

    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path.name} with sequence_path column")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 16-frame numpy sequences from cropped face crops and "
            "add sequence_path column to train/val/test CSVs."
        )
    )
    parser.add_argument(
        "--splits_dir",
        default="data/splits",
        help="Directory containing train/val/test CSV files (default: data/splits)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/sequences",
        help="Output directory for .npy sequence files (default: data/processed/sequences)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"Number of frames per sequence (default: {SEQUENCE_LENGTH})",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [splits_dir / f"{s}.csv" for s in ["train", "val", "test"]]
    found = [p for p in csv_files if p.exists()]

    if not found:
        print(
            f"No CSV files found in {splits_dir}. "
            "Run build_dataset.py first (or run this before build_dataset.py "
            "and pass the faces directory directly with --faces_dir)."
        )
        sys.exit(1)

    print(f"Processing {len(found)} CSV file(s) with sequence_length={args.sequence_length}")
    for csv_path in found:
        print(f"\n{csv_path.name}")
        process_split(csv_path, output_dir, args.sequence_length)

    print("\nAll done. Re-run build_dataset.py if you need sequence_path in a fresh split.")


if __name__ == "__main__":
    main()
