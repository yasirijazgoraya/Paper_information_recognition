"""
Day 1 — Tesseract Baseline on FUNSD

Purpose:
    Run Tesseract OCR on 5 FUNSD test images, compare extracted text against
    ground-truth annotations, and report a baseline F1 score. This is the
    smoke test that proves the end-to-end evaluation pipeline works.

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    python code/layer1_vision/day1_tesseract_funsd.py

Outputs:
    results/extractions/FUNSD/tesseract/<doc_id>.json   (per-doc OCR output)
    results/metrics/day1_tesseract_funsd.csv            (per-doc metrics)
    Console summary with overall F1, precision, recall
"""

import json
import time
from pathlib import Path
from typing import Set

import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
FUNSD_IMAGES = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "images"
FUNSD_ANNOTS = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "annotations"

OUTPUT_DIR = PROJECT_ROOT / "results" / "extractions" / "FUNSD" / "tesseract"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Day 1: smoke test on 5 documents. Set to None to run on all 50.
NUM_DOCS = 50


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_funsd_ground_truth(annot_path: Path) -> Set[str]:
    """
    Load FUNSD annotation JSON and return the set of unique text tokens
    that appear as 'answer' fields (the values an SME would care about).

    FUNSD annotation structure:
        {
          "form": [
            {"text": "Name:", "label": "question", ...},
            {"text": "John Smith", "label": "answer", ...},
            ...
          ]
        }
    """
    with open(annot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = set()
    for item in data.get("form", []):
        text = item.get("text", "").strip()
        if text:
            # Add lowercased tokens for fuzzy comparison
            for token in text.lower().split():
                # Only keep meaningful tokens (alphanumeric, length > 1)
                if len(token) > 1 and any(c.isalnum() for c in token):
                    tokens.add(token)
    return tokens


def run_tesseract(image_path: Path) -> dict:
    """Run Tesseract on a single image. Returns dict with text and timing."""
    image = Image.open(image_path)

    start = time.perf_counter()
    text = pytesseract.image_to_string(image)
    elapsed = time.perf_counter() - start

    return {
        "raw_text": text,
        "tokens_extracted": _tokenize(text),
        "latency_sec": round(elapsed, 3),
    }


def _tokenize(text: str) -> Set[str]:
    """Lowercase + split into meaningful tokens."""
    tokens = set()
    for token in text.lower().split():
        cleaned = "".join(c for c in token if c.isalnum())
        if len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def calculate_metrics(predicted: Set[str], ground_truth: Set[str]) -> dict:
    """
    Token-level Precision, Recall, F1.

    Note: This is a simple smoke-test metric. For the final paper we'll use
    field-level F1 with proper entity matching. For Day 1, token overlap is
    enough to confirm the pipeline is working.
    """
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    # Need to handle the case where predicted tokens dict has set values
    if isinstance(predicted, dict):
        predicted = predicted.get("tokens_extracted", set())

    true_positives = len(predicted & ground_truth)
    false_positives = len(predicted - ground_truth)
    false_negatives = len(ground_truth - predicted)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Setup output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify paths
    if not FUNSD_IMAGES.exists():
        raise FileNotFoundError(f"FUNSD images not found at {FUNSD_IMAGES}")
    if not FUNSD_ANNOTS.exists():
        raise FileNotFoundError(f"FUNSD annotations not found at {FUNSD_ANNOTS}")

    # Verify Tesseract is installed
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

    # Collect images
    image_files = sorted(FUNSD_IMAGES.glob("*.png"))
    if NUM_DOCS:
        image_files = image_files[:NUM_DOCS]

    print(f"Found {len(image_files)} test images to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print("─" * 70)

    # Process each image
    results = []
    for image_path in tqdm(image_files, desc="Processing"):
        doc_id = image_path.stem
        annot_path = FUNSD_ANNOTS / f"{doc_id}.json"

        if not annot_path.exists():
            print(f"⚠️  Missing annotation for {doc_id}, skipping")
            continue

        # Run Tesseract
        ocr_output = run_tesseract(image_path)

        # Load ground truth
        ground_truth = load_funsd_ground_truth(annot_path)

        # Compute metrics
        metrics = calculate_metrics(ocr_output["tokens_extracted"], ground_truth)

        # Save raw OCR output
        output_file = OUTPUT_DIR / f"{doc_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "raw_text": ocr_output["raw_text"],
                "tokens_extracted": sorted(list(ocr_output["tokens_extracted"])),
                "tokens_ground_truth": sorted(list(ground_truth)),
                "latency_sec": ocr_output["latency_sec"],
                "metrics": metrics,
            }, f, indent=2, ensure_ascii=False)

        # Collect for summary table
        results.append({
            "doc_id": doc_id,
            "tokens_predicted": len(ocr_output["tokens_extracted"]),
            "tokens_ground_truth": len(ground_truth),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "latency_sec": ocr_output["latency_sec"],
        })

    # ─────────────────────────────────────────────────────────────────────
    # Save metrics table
    # ─────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    metrics_file = METRICS_DIR / "day1_tesseract_funsd.csv"
    df.to_csv(metrics_file, index=False)

    print("\n" + "═" * 70)
    print("RESULTS — Per-Document")
    print("═" * 70)
    print(df.to_string(index=False))

    print("\n" + "═" * 70)
    print("RESULTS — Summary")
    print("═" * 70)
    print(f"Documents processed: {len(df)}")
    print(f"Mean Precision: {df['precision'].mean():.3f}")
    print(f"Mean Recall:    {df['recall'].mean():.3f}")
    print(f"Mean F1:        {df['f1'].mean():.3f}")
    print(f"Mean Latency:   {df['latency_sec'].mean():.2f} sec/doc")
    print(f"Total time:     {df['latency_sec'].sum():.1f} sec")

    print(f"\n✅ Outputs saved to: {OUTPUT_DIR}")
    print(f"✅ Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
