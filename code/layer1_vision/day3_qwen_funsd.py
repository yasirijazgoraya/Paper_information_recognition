"""
Day 3 — Qwen2-VL-7B-Instruct on FUNSD

Purpose:
    Run Qwen2-VL-7B (multimodal vision-language model) on the full FUNSD test
    set. Compare results to Tesseract (Day 1) and PaddleOCR (Day 2). This is
    the local equivalent of cloud VLMs like Claude Vision and GPT-4o — the
    most important comparison for our research.

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    export HF_HOME=/mnt/yasir_drive/E_DATA/ResearchProject/models/hf_cache
    python code/layer1_vision/day3_qwen_funsd.py

Outputs:
    results/extractions/FUNSD/qwen2vl/<doc_id>.json
    results/metrics/day3_qwen_funsd.csv
    Console summary with 3-way comparison (Tesseract / PaddleOCR / Qwen2-VL)
"""

import json
import os
import time
from pathlib import Path
from typing import Set

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
FUNSD_IMAGES = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "images"
FUNSD_ANNOTS = PROJECT_ROOT / "data" / "dataset" / "testing_data" / "annotations"

OUTPUT_DIR = PROJECT_ROOT / "results" / "extractions" / "FUNSD" / "qwen2vl"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Qwen2-VL-7B-Instruct fits comfortably in 16 GB VRAM with bfloat16
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Run on full FUNSD test set. Set to 5 for a smoke test first if you want.
NUM_DOCS = None

# The prompt — keep it simple and OCR-like for fair comparison
EXTRACT_PROMPT = (
    "Extract all the text content visible in this document image. "
    "Output only the text, preserving the natural reading order top-to-bottom, "
    "left-to-right. Do not add commentary, explanations, or formatting. "
    "Just the raw text."
)

# Image preprocessing: Qwen2-VL handles up to ~1280x720 efficiently
# Larger images use more VRAM but may help accuracy. Tune if OOM.
MAX_PIXELS = 1280 * 720


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (token logic identical to Day 1/2 for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def load_funsd_ground_truth(annot_path: Path) -> Set[str]:
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
    tokens = set()
    for token in text.lower().split():
        cleaned = "".join(c for c in token if c.isalnum())
        if len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def calculate_metrics(predicted: Set[str], ground_truth: Set[str]) -> dict:
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


def run_qwen_on_image(model, processor, image_path: Path) -> dict:
    """Run Qwen2-VL on a single image and return extracted text + timing."""
    image = Image.open(image_path).convert("RGB")

    # Build the chat-format message Qwen2-VL expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": EXTRACT_PROMPT},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # Generate
    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
        )
    elapsed = time.perf_counter() - start

    # Decode — only the new (generated) tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "raw_text": output_text,
        "tokens_extracted": _tokenize(output_text),
        "latency_sec": round(elapsed, 3),
        "output_token_count": len(generated_ids_trimmed[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not FUNSD_IMAGES.exists():
        raise FileNotFoundError(f"FUNSD images not found at {FUNSD_IMAGES}")

    # ─── Confirm HF cache ────────────────────────────────────────────────
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface (default)")
    print(f"HF_HOME: {hf_home}")
    print(f"Device:  {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM:    {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # ─── Load model and processor ────────────────────────────────────────
    print(f"\nLoading {MODEL_ID}...")
    print("First run downloads ~16 GB — be patient.\n")

    load_start = time.perf_counter()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        max_pixels=MAX_PIXELS,
    )
    load_time = time.perf_counter() - load_start
    print(f"✅ Model loaded in {load_time:.1f} sec")

    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"VRAM after load: {used / 1e9:.1f} GB used / {total / 1e9:.1f} GB total\n")

    # ─── Collect images ──────────────────────────────────────────────────
    image_files = sorted(FUNSD_IMAGES.glob("*.png"))
    if NUM_DOCS:
        image_files = image_files[:NUM_DOCS]

    print(f"Processing {len(image_files)} FUNSD test images")
    print(f"Output: {OUTPUT_DIR}")
    print("─" * 70)

    # ─── Inference loop ──────────────────────────────────────────────────
    results = []
    for image_path in tqdm(image_files, desc="Qwen2-VL"):
        doc_id = image_path.stem
        annot_path = FUNSD_ANNOTS / f"{doc_id}.json"

        if not annot_path.exists():
            print(f"⚠️  Missing annotation for {doc_id}, skipping")
            continue

        try:
            vlm_output = run_qwen_on_image(model, processor, image_path)
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️  OOM on {doc_id}, clearing cache and skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"⚠️  Error on {doc_id}: {e}")
            continue

        ground_truth = load_funsd_ground_truth(annot_path)
        metrics = calculate_metrics(vlm_output["tokens_extracted"], ground_truth)

        # Save per-doc output
        out_file = OUTPUT_DIR / f"{doc_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "model": MODEL_ID,
                "prompt": EXTRACT_PROMPT,
                "raw_text": vlm_output["raw_text"],
                "tokens_extracted": sorted(list(vlm_output["tokens_extracted"])),
                "tokens_ground_truth": sorted(list(ground_truth)),
                "latency_sec": vlm_output["latency_sec"],
                "output_token_count": vlm_output["output_token_count"],
                "metrics": metrics,
            }, f, indent=2, ensure_ascii=False)

        results.append({
            "doc_id": doc_id,
            "tokens_predicted": len(vlm_output["tokens_extracted"]),
            "tokens_ground_truth": len(ground_truth),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "latency_sec": vlm_output["latency_sec"],
            "output_tokens": vlm_output["output_token_count"],
        })

    # ─── Summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    metrics_file = METRICS_DIR / "day3_qwen_funsd.csv"
    df.to_csv(metrics_file, index=False)

    print("\n" + "═" * 70)
    print("RESULTS — Summary (Qwen2-VL-7B)")
    print("═" * 70)
    print(f"Documents processed: {len(df)}")
    print(f"Mean Precision:      {df['precision'].mean():.3f}")
    print(f"Mean Recall:         {df['recall'].mean():.3f}")
    print(f"Mean F1:             {df['f1'].mean():.3f}")
    print(f"Mean Latency:        {df['latency_sec'].mean():.2f} sec/doc")
    print(f"Total inference:     {df['latency_sec'].sum():.1f} sec")

    # ─── Three-way comparison ────────────────────────────────────────────
    tess_csv = METRICS_DIR / "day1_tesseract_funsd.csv"
    paddle_csv = METRICS_DIR / "day2_paddleocr_funsd.csv"

    if tess_csv.exists() and paddle_csv.exists():
        tess_df = pd.read_csv(tess_csv)
        paddle_df = pd.read_csv(paddle_csv)

        print("\n" + "═" * 70)
        print("THREE-WAY COMPARISON — FUNSD Test Set (50 documents)")
        print("═" * 70)
        print(f"{'Metric':<22} {'Tesseract':<14} {'PaddleOCR':<14} {'Qwen2-VL-7B':<14}")
        print("─" * 70)
        for metric in ["precision", "recall", "f1"]:
            tess_val = tess_df[metric].mean()
            paddle_val = paddle_df[metric].mean()
            qwen_val = df[metric].mean()
            print(f"{metric.capitalize():<22} {tess_val:<14.3f} {paddle_val:<14.3f} {qwen_val:<14.3f}")

        tess_lat = tess_df['latency_sec'].mean()
        paddle_lat = paddle_df['latency_sec'].mean()
        qwen_lat = df['latency_sec'].mean()
        print(f"{'Latency (sec/doc)':<22} {tess_lat:<14.3f} {paddle_lat:<14.3f} {qwen_lat:<14.3f}")

        print("\nKey takeaway:")
        print(f"  Best F1:     {max(['Tesseract', 'PaddleOCR', 'Qwen2-VL'][i] for i, v in enumerate([tess_df['f1'].mean(), paddle_df['f1'].mean(), df['f1'].mean()]) if v == max(tess_df['f1'].mean(), paddle_df['f1'].mean(), df['f1'].mean()))}")
        print(f"  Fastest:     {min(['Tesseract', 'PaddleOCR', 'Qwen2-VL'][i] for i, v in enumerate([tess_lat, paddle_lat, qwen_lat]) if v == min(tess_lat, paddle_lat, qwen_lat))}")

    print(f"\n✅ Outputs:  {OUTPUT_DIR}")
    print(f"✅ Metrics:  {metrics_file}")


if __name__ == "__main__":
    main()
