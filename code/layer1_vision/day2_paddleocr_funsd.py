"""
Day 2 — PaddleOCR (Layout-Aware) on FUNSD

Purpose:
    Run PaddleOCR on the full FUNSD test set (50 documents), compute the same
    token-overlap F1 metric used for Tesseract in Day 1, and compare results.
    PaddleOCR is layout-aware, so we expect a meaningful F1 improvement.

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    python code/layer1_vision/day2_paddleocr_funsd.py

Outputs:
    results/extractions/FUNSD/paddleocr/<doc_id>.json   (per-doc OCR output)
    results/metrics/day2_paddleocr_funsd.csv            (per-doc metrics)
    Console summary with overall F1, precision, recall + Tesseract delta
"""

import json
import time
import warnings
from pathlib import Path
from typing import Set

import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

# PaddleOCR is chatty on first load; suppress unless something matters
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
FUNSD_IMAGES = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "images"
FUNSD_ANNOTS = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "annotations"

OUTPUT_DIR = PROJECT_ROOT / "results" / "extractions" / "FUNSD" / "paddleocr"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Run on full FUNSD test set
NUM_DOCS = None  # set to 5 for a smoke test, None for all 50

# PaddleOCR config — English, with angle correction for tilted scans
USE_GPU = True   # RTX 4080 SUPER — let's use it
LANG = "en"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (identical to Day 1 for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def load_funsd_ground_truth(annot_path: Path) -> Set[str]:
    """Load FUNSD annotation and extract token set (same logic as Day 1)."""
    with open(annot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = set()
    for item in data.get("form", []):
        text = item.get("text", "").strip()
        if text:
            for token in text.lower().split():
                if len(token) > 1 and any(c.isalnum() for c in token):
                    tokens.add(token)
    return tokens


def _tokenize(text: str) -> Set[str]:
    """Lowercase + clean tokens (same logic as Day 1)."""
    tokens = set()
    for token in text.lower().split():
        cleaned = "".join(c for c in token if c.isalnum())
        if len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def run_paddleocr(ocr_engine: PaddleOCR, image_path: Path) -> dict:
    """Run PaddleOCR on one image. Returns text, tokens, boxes, and timing."""
    start = time.perf_counter()
    result = ocr_engine.ocr(str(image_path), cls=True)
    elapsed = time.perf_counter() - start

    # PaddleOCR returns a nested structure:
    #   result = [[ [box, (text, confidence)], [box, (text, confidence)], ... ]]
    # We flatten to extract just the text strings + boxes + confidences.
    text_segments = []
    boxes = []
    confidences = []

    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text, conf = line[1]
            text_segments.append(text)
            boxes.append(box)
            confidences.append(float(conf))

    full_text = " ".join(text_segments)
    tokens = _tokenize(full_text)

    return {
        "raw_text": full_text,
        "text_segments": text_segments,
        "boxes": boxes,
        "confidences": confidences,
        "tokens_extracted": tokens,
        "latency_sec": round(elapsed, 3),
    }


def calculate_metrics(predicted: Set[str], ground_truth: Set[str]) -> dict:
    """Token-level Precision, Recall, F1 (identical to Day 1)."""
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not FUNSD_IMAGES.exists():
        raise FileNotFoundError(f"FUNSD images not found at {FUNSD_IMAGES}")

    # ─── Initialise PaddleOCR ────────────────────────────────────────────
    print("Initialising PaddleOCR (this takes ~20 sec on first run)...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,   # detect rotated text
        lang=LANG,
        use_gpu=False,
        show_log=False,
    )
    print("✅ PaddleOCR ready\n")

    # ─── Collect images ──────────────────────────────────────────────────
    image_files = sorted(FUNSD_IMAGES.glob("*.png"))
    if NUM_DOCS:
        image_files = image_files[:NUM_DOCS]

    print(f"Processing {len(image_files)} FUNSD test images")
    print(f"Output: {OUTPUT_DIR}")
    print("─" * 70)

    # ─── Run OCR ─────────────────────────────────────────────────────────
    results = []
    for image_path in tqdm(image_files, desc="PaddleOCR"):
        doc_id = image_path.stem
        annot_path = FUNSD_ANNOTS / f"{doc_id}.json"

        if not annot_path.exists():
            print(f"⚠️  Missing annotation for {doc_id}, skipping")
            continue

        ocr_output = run_paddleocr(ocr_engine, image_path)
        ground_truth = load_funsd_ground_truth(annot_path)
        metrics = calculate_metrics(ocr_output["tokens_extracted"], ground_truth)

        # Save per-doc output
        out_file = OUTPUT_DIR / f"{doc_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "raw_text": ocr_output["raw_text"],
                "text_segments": ocr_output["text_segments"],
                "boxes": ocr_output["boxes"],
                "confidences": ocr_output["confidences"],
                "tokens_extracted": sorted(list(ocr_output["tokens_extracted"])),
                "tokens_ground_truth": sorted(list(ground_truth)),
                "latency_sec": ocr_output["latency_sec"],
                "metrics": metrics,
            }, f, indent=2, ensure_ascii=False)

        results.append({
            "doc_id": doc_id,
            "tokens_predicted": len(ocr_output["tokens_extracted"]),
            "tokens_ground_truth": len(ground_truth),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "latency_sec": ocr_output["latency_sec"],
            "mean_confidence": round(sum(ocr_output["confidences"]) / len(ocr_output["confidences"]), 3)
                if ocr_output["confidences"] else 0.0,
        })

    # ─── Summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    metrics_file = METRICS_DIR / "day2_paddleocr_funsd.csv"
    df.to_csv(metrics_file, index=False)

    print("\n" + "═" * 70)
    print("RESULTS — Per-Document")
    print("═" * 70)
    print(df.to_string(index=False))

    print("\n" + "═" * 70)
    print("RESULTS — Summary (PaddleOCR)")
    print("═" * 70)
    print(f"Documents processed: {len(df)}")
    print(f"Mean Precision:      {df['precision'].mean():.3f}")
    print(f"Mean Recall:         {df['recall'].mean():.3f}")
    print(f"Mean F1:             {df['f1'].mean():.3f}")
    print(f"Mean Latency:        {df['latency_sec'].mean():.2f} sec/doc")
    print(f"Total time:          {df['latency_sec'].sum():.1f} sec")
    print(f"Mean Confidence:     {df['mean_confidence'].mean():.3f}")

    # ─── Compare to Tesseract baseline ───────────────────────────────────
    tesseract_csv = METRICS_DIR / "day1_tesseract_funsd.csv"
    if tesseract_csv.exists():
        tess_df = pd.read_csv(tesseract_csv)

        print("\n" + "═" * 70)
        print("COMPARISON — PaddleOCR vs Tesseract (Day 1)")
        print("═" * 70)
        print(f"{'Metric':<20} {'Tesseract':<15} {'PaddleOCR':<15} {'Delta':<10}")
        print("─" * 60)
        for metric in ["precision", "recall", "f1"]:
            tess_val = tess_df[metric].mean()
            paddle_val = df[metric].mean()
            delta = paddle_val - tess_val
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{metric.capitalize():<20} {tess_val:<15.3f} {paddle_val:<15.3f} {delta:+.3f} {arrow}")

        # Latency
        tess_lat = tess_df['latency_sec'].mean()
        paddle_lat = df['latency_sec'].mean()
        print(f"{'Latency (sec/doc)':<20} {tess_lat:<15.3f} {paddle_lat:<15.3f} {paddle_lat - tess_lat:+.3f}")

    print(f"\n✅ Outputs:  {OUTPUT_DIR}")
    print(f"✅ Metrics:  {metrics_file}")


if __name__ == "__main__":
    main()
