"""Build train/val/test CSV splits from processed faces and FFT images.

Dataset roles (per proposal):
  FF++       → train + val + test (the only training source)
  Celeb-DF v2 → evaluation-only cross-dataset generalization test
  DFDC Preview → evaluation-only cross-dataset generalization test

Outputs:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
  data/splits/celebdf_eval.csv   (cross-dataset test set 1)
  data/splits/dfdc_eval.csv      (cross-dataset test set 2)

Each CSV has columns:
  frame_path, fft_path, label, source, manipulation_type, split
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ── label / metadata inference ────────────────────────────────────────────────

SOURCE_MAP = {
    "faceforensics": "faceforensics",
    "ff++": "faceforensics",
    "celebdf": "celebdf",
    "celeb-df": "celebdf",
    "celeb_df": "celebdf",
    "dfdc": "dfdc",
}

MANIPULATION_MAP = {
    "deepfakes": "deepfakes",
    "face2face": "face2face",
    "faceswap": "faceswap",
    "neuraltextures": "neuraltextures",
}

# Sources excluded from train/val/test; each gets its own eval CSV.
EVAL_ONLY_SOURCES = {"celebdf", "dfdc"}


def infer_label(path_str: str) -> int:
    lower = path_str.lower()
    return 0 if ("real" in lower or "original" in lower) else 1


def infer_source(path_str: str) -> str:
    lower = path_str.lower()
    for keyword, source in SOURCE_MAP.items():
        if keyword in lower:
            return source
    return "unknown"


def infer_manipulation(path_str: str) -> str:
    lower = path_str.lower()
    for keyword, manip in MANIPULATION_MAP.items():
        if keyword in lower:
            return manip
    return "unknown"


def video_id_from_path(face_path: Path, faces_dir: Path) -> str:
    """Parent subdirectory path as the video-level key (one dir per video)."""
    return str(face_path.relative_to(faces_dir).parent)


# ── main ──────────────────────────────────────────────────────────────────────

def build_dataset(faces_dir: str, fft_dir: str, output_dir: str) -> None:
    faces_dir = Path(faces_dir)
    fft_dir = Path(fft_dir)
    output_dir = Path(output_dir)

    if not faces_dir.exists():
        print(f"ERROR: faces_dir does not exist: {faces_dir}")
        sys.exit(1)
    if not fft_dir.exists():
        print(f"ERROR: fft_dir does not exist: {fft_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    face_files = list(faces_dir.rglob("*.jpg")) + list(faces_dir.rglob("*.jpeg"))
    print(f"Found {len(face_files)} face image(s)")

    rows = []
    missing_fft = 0

    for face_path in tqdm(face_files, desc="Scanning pairs"):
        rel_path = face_path.relative_to(faces_dir)
        fft_path = fft_dir / rel_path

        if not fft_path.exists():
            missing_fft += 1
            continue

        path_str = str(face_path)
        rows.append(
            {
                "frame_path": str(face_path),
                "fft_path": str(fft_path),
                "label": infer_label(path_str),
                "source": infer_source(path_str),
                "manipulation_type": infer_manipulation(path_str),
                "video_id": video_id_from_path(face_path, faces_dir),
                "split": "",
            }
        )

    if missing_fft:
        print(f"Warning: {missing_fft} face image(s) had no matching FFT file — skipped")

    if not rows:
        print("No paired samples found. Nothing to write.")
        return

    df = pd.DataFrame(rows)
    print(f"Total paired samples: {len(df)}")

    # ── separate eval-only sources ────────────────────────────────────────────
    eval_mask = df["source"].isin(EVAL_ONLY_SOURCES)
    eval_df = df[eval_mask].copy()
    train_df = df[~eval_mask].copy()

    # Write one CSV per eval-only source
    for source_name in eval_df["source"].unique():
        source_df = eval_df[eval_df["source"] == source_name].drop(
            columns=["video_id", "split"]
        )
        source_df = source_df.assign(split="eval")
        out_path = output_dir / f"{source_name}_eval.csv"
        source_df.to_csv(out_path, index=False)
        real = (source_df["label"] == 0).sum()
        fake = (source_df["label"] == 1).sum()
        print(
            f"Saved {len(source_df):>6} rows → {out_path}  "
            f"(real: {real}, fake: {fake})  [cross-dataset eval only]"
        )

    if train_df.empty:
        print("No FF++ samples found for train/val/test splits.")
        return

    # ── video-level stratified split on training sources (FF++ only) ─────────
    video_meta = train_df.groupby("video_id")["label"].first().reset_index()

    train_vids, temp_vids = train_test_split(
        video_meta["video_id"],
        test_size=0.30,
        stratify=video_meta["label"],
        random_state=42,
    )
    temp_meta = video_meta[video_meta["video_id"].isin(temp_vids)]
    val_vids, test_vids = train_test_split(
        temp_meta["video_id"],
        test_size=0.50,
        stratify=temp_meta["label"],
        random_state=42,
    )

    train_df["split"] = "train"
    train_df.loc[train_df["video_id"].isin(val_vids), "split"] = "val"
    train_df.loc[train_df["video_id"].isin(test_vids), "split"] = "test"

    df_out = train_df.drop(columns=["video_id"])

    for split_name in ["train", "val", "test"]:
        split_df = df_out[df_out["split"] == split_name]
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        real = (split_df["label"] == 0).sum()
        fake = (split_df["label"] == 1).sum()
        print(
            f"Saved {len(split_df):>6} rows → {out_path}  (real: {real}, fake: {fake})"
        )

    print("\nLabel distribution per split (FF++ only):")
    print(df_out.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test CSV splits (FF++ only) and separate eval CSVs "
            "for cross-dataset generalization testing (Celeb-DF v2, DFDC Preview)."
        )
    )
    parser.add_argument("--faces_dir", required=True, help="Directory containing cropped face images")
    parser.add_argument("--fft_dir", required=True, help="Directory containing FFT images")
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="Output directory for CSV files (default: data/splits)",
    )
    args = parser.parse_args()
    build_dataset(args.faces_dir, args.fft_dir, args.output_dir)


if __name__ == "__main__":
    main()
