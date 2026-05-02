import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


IMBALANCE_THRESHOLD = 0.60  # warn if either class exceeds this fraction


# ── loading ──────────────────────────────────────────────────────────────────

def load_splits(splits_dir: str) -> dict[str, pd.DataFrame]:
    splits_dir = Path(splits_dir)
    dfs: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        path = splits_dir / f"{split}.csv"
        if path.exists():
            dfs[split] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found — skipping")
    return dfs


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(dfs: dict[str, pd.DataFrame]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("DATASET SUMMARY")
    print(sep)

    for split_name, df in dfs.items():
        total = len(df)
        real = (df["label"] == 0).sum()
        fake = (df["label"] == 1).sum()
        real_pct = real / total if total else 0
        fake_pct = fake / total if total else 0

        print(f"\n{split_name.upper()} — {total:,} total frames")
        print(f"  Real: {real:,} ({100*real_pct:.1f}%)   Fake: {fake:,} ({100*fake_pct:.1f}%)")

        if real_pct > IMBALANCE_THRESHOLD or fake_pct > IMBALANCE_THRESHOLD:
            print("  *** WARNING: real/fake ratio exceeds 60/40 — consider rebalancing ***")

        if "source" in df.columns:
            print("  Source breakdown:")
            for source, count in df["source"].value_counts().items():
                print(f"    {source:<20} {count:,}")

    print(f"\n{sep}\n")


# ── file existence check ──────────────────────────────────────────────────────

def check_file_existence(dfs: dict[str, pd.DataFrame]) -> None:
    print("Checking file existence on disk...")
    total_missing = 0

    for split_name, df in dfs.items():
        missing_frames = (~df["frame_path"].map(os.path.exists)).sum()
        missing_fft = (~df["fft_path"].map(os.path.exists)).sum()

        if missing_frames:
            print(f"  {split_name}: {missing_frames} missing frame_path file(s)")
            total_missing += missing_frames
        if missing_fft:
            print(f"  {split_name}: {missing_fft} missing fft_path file(s)")
            total_missing += missing_fft

    if total_missing == 0:
        print("  All files present on disk.\n")
    else:
        print(f"  Total missing: {total_missing}\n")


# ── visual sample display ─────────────────────────────────────────────────────

def _load_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_gray(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def show_sample_images(dfs: dict[str, pd.DataFrame], n: int = 5) -> None:
    all_df = pd.concat(dfs.values(), ignore_index=True)
    real_df = all_df[all_df["label"] == 0]
    fake_df = all_df[all_df["label"] == 1]

    n_real = min(n, len(real_df))
    n_fake = min(n, len(fake_df))

    if n_real == 0 or n_fake == 0:
        print("Not enough samples to display comparison grid — skipping visualization")
        return

    real_rows = real_df.sample(n_real, random_state=42).to_dict("records")
    fake_rows = fake_df.sample(n_fake, random_state=42).to_dict("records")

    # ── face crops ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle("Face Crops — Real (top row) vs Fake (bottom row)", fontsize=13)
    for i in range(n):
        for row_idx, rows in enumerate([real_rows, fake_rows]):
            ax = axes[row_idx][i]
            ax.axis("off")
            if i < len(rows):
                img = _load_bgr(rows[i]["frame_path"])
                if img is not None:
                    ax.imshow(img)
                    ax.set_title("Real" if row_idx == 0 else "Fake", fontsize=9)
    plt.tight_layout()
    out_face = "results/sample_faces.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(out_face, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"Face grid saved → {out_face}")

    # ── FFT spectra ───────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, n, figsize=(3 * n, 6))
    fig2.suptitle("FFT Spectra — Real (top row) vs Fake (bottom row)", fontsize=13)
    for i in range(n):
        for row_idx, rows in enumerate([real_rows, fake_rows]):
            ax = axes2[row_idx][i]
            ax.axis("off")
            if i < len(rows):
                img = _load_gray(rows[i]["fft_path"])
                if img is not None:
                    ax.imshow(img, cmap="hot")
                    ax.set_title("Real" if row_idx == 0 else "Fake", fontsize=9)
    plt.tight_layout()
    out_fft = "results/sample_fft.png"
    plt.savefig(out_fft, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"FFT grid saved → {out_fft}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify dataset CSV splits and display sample images."
    )
    parser.add_argument(
        "--splits_dir",
        default="data/splits",
        help="Directory containing train/val/test CSV files (default: data/splits)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to show per class in the visual grids (default: 5)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Skip matplotlib display (still saves PNG files)",
    )
    args = parser.parse_args()

    dfs = load_splits(args.splits_dir)
    if not dfs:
        print("No CSV files found. Run build_dataset.py first.")
        sys.exit(1)

    print_summary(dfs)
    check_file_existence(dfs)

    if not args.no_display:
        try:
            show_sample_images(dfs, args.n_samples)
        except Exception as e:
            print(f"Could not display images: {e}")


if __name__ == "__main__":
    main()
