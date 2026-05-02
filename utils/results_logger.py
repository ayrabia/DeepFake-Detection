"""Log experiment results to results/ as timestamped JSON files.

Usage (from any notebook or script):
    from utils.results_logger import log_results

    log_results('spatial_cnn', {
        'val_auc':  0.9821,
        'test_auc': 0.9743,
        'per_manipulation': {
            'deepfakes':       0.991,
            'face2face':       0.963,
            'faceswap':        0.978,
            'neuraltextures':  0.961,
        },
        'celebdf_auc':  0.812,
        'dfdc_auc':     0.734,
        'epochs':  20,
        'batch_size': 64,
    })

Produces: results/spatial_cnn_20260501_143022.json
Also maintains results/summary.json with one entry per model (latest run only).
"""

import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def log_results(model_name: str, metrics: dict, results_dir: str | None = None) -> str:
    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "model":     model_name,
        "timestamp": timestamp,
        "metrics":   metrics,
    }

    # Individual timestamped file
    individual_path = out_dir / f"{model_name}_{timestamp}.json"
    with open(individual_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Rolling summary (one entry per model_name, latest run wins)
    summary_path = out_dir / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                summary = {}

    summary[model_name] = payload
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved → {individual_path}")
    print(f"Summary updated → {summary_path}")
    return str(individual_path)


def load_summary(results_dir: str | None = None) -> dict:
    """Return the summary dict keyed by model name."""
    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)
