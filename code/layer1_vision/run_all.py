"""
Run All — Layer 1 Full Evaluation Matrix

Executes the unified runner across all three datasets (FUNSD, SROIE, CORD) with
all three models (Tesseract, PaddleOCR, Qwen2-VL). Produces a single
9-cell comparison table at the end (3 datasets x 3 models).

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    export HF_HOME=/mnt/yasir_drive/E_DATA/ResearchProject/models/hf_cache

    # Full run (everything)
    python code/layer1_vision/run_all.py

    # Smoke test (5 docs per dataset)
    python code/layer1_vision/run_all.py --limit 5

    # Skip datasets already done
    python code/layer1_vision/run_all.py --skip FUNSD

Outputs:
    results/metrics/<dataset>_<model>.csv          (per-doc metrics, 9 files)
    results/metrics/<dataset>_comparison.csv       (per-dataset summary, 3 files)
    results/metrics/full_comparison.csv            (single 9-cell matrix)
    results/metrics/full_comparison.md             (markdown table for paper)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
RUNNER = PROJECT_ROOT / "code" / "layer1_vision" / "run_layer1.py"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DATASETS = ["FUNSD", "SROIE", "CORD"]
MODELS   = ["tesseract", "paddleocr", "qwen2vl"]


def run_dataset(dataset: str, limit: int | None) -> bool:
    """Invoke run_layer1.py for one dataset. Returns True on success."""
    print(f"\n{'#'*72}\n# {dataset}\n{'#'*72}")
    cmd = [sys.executable, str(RUNNER), "--dataset", dataset]
    if limit:
        cmd += ["--limit", str(limit)]
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        print(f"\n✅ {dataset} done in {elapsed/60:.1f} min")
        return True
    print(f"\n❌ {dataset} failed (exit {result.returncode})")
    return False


def build_full_comparison() -> pd.DataFrame:
    """Aggregate the 9 per-doc CSVs into a single matrix."""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            csv = METRICS_DIR / f"{dataset.lower()}_{model}.csv"
            if not csv.exists():
                print(f"⚠️  Missing: {csv.name}")
                continue
            df = pd.read_csv(csv)
            rows.append({
                "dataset":       dataset,
                "model":         model,
                "documents":     len(df),
                "precision":     round(df["precision"].mean(), 3),
                "recall":        round(df["recall"].mean(), 3),
                "f1":            round(df["f1"].mean(), 3),
                "latency_sec":   round(df["latency_sec"].mean(), 2),
                "f1_std":        round(df["f1"].std(), 3),
            })
    return pd.DataFrame(rows)


def render_markdown_table(df: pd.DataFrame) -> str:
    """Build a paper-ready markdown table grouped by dataset."""
    lines = ["# Layer 1 — Full Evaluation Matrix\n",
             "| Dataset | Model | Docs | Precision | Recall | F1 | F1 SD | Latency (s) |",
             "|---------|-------|------|-----------|--------|----|----|----|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['documents']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['f1_std']:.3f} | {r['latency_sec']:.2f} |"
        )

    lines.append("\n## Best F1 per dataset\n")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if len(sub):
            best = sub.loc[sub["f1"].idxmax()]
            lines.append(f"- **{ds}**: {best['model']} (F1 = {best['f1']:.3f})")

    lines.append("\n## Best model overall (mean F1 across datasets)\n")
    by_model = df.groupby("model")["f1"].mean().sort_values(ascending=False)
    for model, f1 in by_model.items():
        lines.append(f"- **{model}**: mean F1 = {f1:.3f}")

    return "\n".join(lines) + "\n"


def render_console_table(df: pd.DataFrame):
    """Pretty 9-cell summary printed at end of run."""
    print("\n" + "═"*78)
    print(" " * 24 + "LAYER 1 — FULL COMPARISON")
    print("═"*78)
    print(f"{'Dataset':<10} {'Model':<12} {'Docs':>5} {'Prec':>7} {'Recall':>7} "
          f"{'F1':>7} {'F1 SD':>7} {'Latency':>8}")
    print("─"*78)
    current_dataset = None
    for _, r in df.iterrows():
        if r["dataset"] != current_dataset:
            current_dataset = r["dataset"]
            sep = "·"*78
            if not df.iloc[0]["dataset"] == current_dataset:
                print(sep)
        print(f"{r['dataset']:<10} {r['model']:<12} {r['documents']:>5} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} "
              f"{r['f1_std']:>7.3f} {r['latency_sec']:>7.2f}s")
    print("═"*78)

    # Best per dataset
    print("\nBest F1 per dataset:")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if len(sub):
            best = sub.loc[sub["f1"].idxmax()]
            print(f"  {ds:<8} → {best['model']:<12} (F1 = {best['f1']:.3f})")

    # Best model overall
    print("\nMean F1 across all datasets:")
    by_model = df.groupby("model")["f1"].mean().sort_values(ascending=False)
    for model, f1 in by_model.items():
        print(f"  {model:<12} mean F1 = {f1:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Docs per dataset (smoke test)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Datasets to skip (e.g. --skip FUNSD)")
    ap.add_argument("--only-summary", action="store_true",
                    help="Skip running, just rebuild summary from existing CSVs")
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()

    if not args.only_summary:
        for dataset in DATASETS:
            if dataset in args.skip:
                print(f"\n⏭️  Skipping {dataset}")
                continue
            run_dataset(dataset, args.limit)

    # Build the unified matrix
    print(f"\n{'#'*72}\n# Building full comparison\n{'#'*72}")
    full = build_full_comparison()
    if full.empty:
        print("❌ No results found. Did the runs complete?")
        return

    csv_path = METRICS_DIR / "full_comparison.csv"
    md_path  = METRICS_DIR / "full_comparison.md"
    full.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown_table(full), encoding="utf-8")

    render_console_table(full)

    total = time.perf_counter() - overall_start
    print(f"\n✅ Total runtime: {total/60:.1f} minutes")
    print(f"✅ CSV: {csv_path}")
    print(f"✅ Markdown (paper-ready): {md_path}")


if __name__ == "__main__":
    main()
