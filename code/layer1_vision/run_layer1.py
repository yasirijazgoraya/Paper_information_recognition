"""
Layer 1 Unified Evaluation Runner

Runs three vision models (Tesseract, PaddleOCR, Qwen2-VL-7B-4bit) on a chosen
dataset (FUNSD, SROIE, or CORD). Same token-level F1 metric as Day 1-3 scripts
so results are fully comparable.

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    export HF_HOME=/mnt/yasir_drive/E_DATA/ResearchProject/models/hf_cache

    # Run all three models on SROIE
    python code/layer1_vision/run_layer1.py --dataset SROIE

    # Run only specific models
    python code/layer1_vision/run_layer1.py --dataset CORD --models tesseract paddleocr

    # Quick smoke test on 5 images
    python code/layer1_vision/run_layer1.py --dataset SROIE --limit 5

Outputs:
    results/extractions/<DATASET>/<model>/<doc_id>.json
    results/metrics/<dataset>_<model>.csv
    results/metrics/<dataset>_comparison.csv
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Set, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTIONS_ROOT = PROJECT_ROOT / "results" / "extractions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS — each returns list of (doc_id, image_path, ground_truth_tokens)
# ═════════════════════════════════════════════════════════════════════════════

def load_funsd() -> List[Tuple[str, Path, Set[str]]]:
    """FUNSD: form JSONs with 'form' list of items with 'text' field."""
    img_dir = DATA_ROOT / "dataset" / "testing_data" / "images"
    ann_dir = DATA_ROOT / "dataset" / "testing_data" / "annotations"

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        doc_id = img_path.stem
        ann_path = ann_dir / f"{doc_id}.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokens = set()
        for item in data.get("form", []):
            tokens |= _tokenize(item.get("text", ""))
        samples.append((doc_id, img_path, tokens))
    return samples


def load_sroie() -> List[Tuple[str, Path, Set[str]]]:
    """SROIE: TXT files with JSON content (company/date/address/total)."""
    img_dir = DATA_ROOT / "SROIE2019" / "test" / "img"
    ann_dir = DATA_ROOT / "SROIE2019" / "test" / "entities"

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        doc_id = img_path.stem
        ann_path = ann_dir / f"{doc_id}.txt"
        if not ann_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        tokens = set()
        for value in data.values():
            tokens |= _tokenize(str(value))
        samples.append((doc_id, img_path, tokens))
    return samples


def load_cord() -> List[Tuple[str, Path, Set[str]]]:
    """CORD: JSON with valid_line -> words -> text."""
    img_dir = DATA_ROOT / "CORD" / "CORD" / "test" / "image"
    ann_dir = DATA_ROOT / "CORD" / "CORD" / "test" / "json"

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        doc_id = img_path.stem
        ann_path = ann_dir / f"{doc_id}.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens = set()
        for line in data.get("valid_line", []):
            for word in line.get("words", []):
                tokens |= _tokenize(word.get("text", ""))
        samples.append((doc_id, img_path, tokens))
    return samples


DATASET_LOADERS = {
    "FUNSD": load_funsd,
    "SROIE": load_sroie,
    "CORD":  load_cord,
}


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN UTILITIES (identical across all scripts)
# ═════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> Set[str]:
    tokens = set()
    for token in str(text).lower().split():
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
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_tesseract(samples, dataset_name: str, output_dir: Path):
    import pytesseract
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    results = []
    for doc_id, img_path, gt_tokens in tqdm(samples, desc="Tesseract"):
        image = Image.open(img_path)
        start = time.perf_counter()
        text = pytesseract.image_to_string(image)
        latency = time.perf_counter() - start
        pred_tokens = _tokenize(text)
        m = calculate_metrics(pred_tokens, gt_tokens)
        _save_extraction(output_dir, doc_id, {
            "raw_text": text, "tokens_extracted": sorted(pred_tokens),
            "tokens_ground_truth": sorted(gt_tokens),
            "latency_sec": round(latency, 3), "metrics": m,
        })
        results.append({"doc_id": doc_id, "tokens_predicted": len(pred_tokens),
                        "tokens_ground_truth": len(gt_tokens),
                        "precision": m["precision"], "recall": m["recall"],
                        "f1": m["f1"], "latency_sec": round(latency, 3)})
    return results


def run_paddleocr(samples, dataset_name: str, output_dir: Path):
    from paddleocr import PaddleOCR
    print("Initialising PaddleOCR (CPU)...")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    results = []
    for doc_id, img_path, gt_tokens in tqdm(samples, desc="PaddleOCR"):
        start = time.perf_counter()
        try:
            out = ocr.ocr(str(img_path), cls=True)
        except Exception as e:
            print(f"⚠️  {doc_id}: {e}")
            continue
        latency = time.perf_counter() - start

        segments, boxes, confs = [], [], []
        if out and out[0]:
            for line in out[0]:
                box = line[0]
                text, conf = line[1]
                segments.append(text); boxes.append(box); confs.append(float(conf))

        full_text = " ".join(segments)
        pred_tokens = _tokenize(full_text)
        m = calculate_metrics(pred_tokens, gt_tokens)
        _save_extraction(output_dir, doc_id, {
            "raw_text": full_text, "text_segments": segments,
            "boxes": boxes, "confidences": confs,
            "tokens_extracted": sorted(pred_tokens),
            "tokens_ground_truth": sorted(gt_tokens),
            "latency_sec": round(latency, 3), "metrics": m,
        })
        results.append({"doc_id": doc_id, "tokens_predicted": len(pred_tokens),
                        "tokens_ground_truth": len(gt_tokens),
                        "precision": m["precision"], "recall": m["recall"],
                        "f1": m["f1"], "latency_sec": round(latency, 3),
                        "mean_confidence": round(sum(confs)/len(confs), 3) if confs else 0.0})
    return results


def run_qwen2vl(samples, dataset_name: str, output_dir: Path):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    PROMPT = ("Extract all the text content visible in this document image. "
              "Output only the text, preserving the natural reading order top-to-bottom, "
              "left-to-right. Do not add commentary, explanations, or formatting. Just the raw text.")
    MAX_PIXELS = 1280 * 720

    print(f"Loading {MODEL_ID} (4-bit quantized)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    free, total = torch.cuda.mem_get_info()
    print(f"✅ Loaded. VRAM: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB\n")

    results = []
    for doc_id, img_path, gt_tokens in tqdm(samples, desc="Qwen2-VL"):
        image = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_in], images=[image], padding=True,
                           return_tensors="pt").to("cuda")

        start = time.perf_counter()
        try:
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=1024,
                                         do_sample=False, temperature=1.0)
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️  OOM on {doc_id}, skipping")
            torch.cuda.empty_cache()
            continue
        latency = time.perf_counter() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
        pred_tokens = _tokenize(output_text)
        m = calculate_metrics(pred_tokens, gt_tokens)
        _save_extraction(output_dir, doc_id, {
            "model": MODEL_ID, "prompt": PROMPT,
            "raw_text": output_text,
            "tokens_extracted": sorted(pred_tokens),
            "tokens_ground_truth": sorted(gt_tokens),
            "latency_sec": round(latency, 3),
            "output_token_count": len(trimmed[0]), "metrics": m,
        })
        results.append({"doc_id": doc_id, "tokens_predicted": len(pred_tokens),
                        "tokens_ground_truth": len(gt_tokens),
                        "precision": m["precision"], "recall": m["recall"],
                        "f1": m["f1"], "latency_sec": round(latency, 3),
                        "output_tokens": len(trimmed[0])})
    # Free VRAM
    del model
    torch.cuda.empty_cache()
    return results


MODEL_RUNNERS = {
    "tesseract": run_tesseract,
    "paddleocr": run_paddleocr,
    "qwen2vl":   run_qwen2vl,
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _save_extraction(out_dir: Path, doc_id: str, data: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, **data}, f, indent=2, ensure_ascii=False)


def _summary(results: List[dict], model_name: str):
    df = pd.DataFrame(results)
    print(f"\n── {model_name.upper()} ── {len(df)} docs")
    print(f"   Mean Precision: {df['precision'].mean():.3f}")
    print(f"   Mean Recall:    {df['recall'].mean():.3f}")
    print(f"   Mean F1:        {df['f1'].mean():.3f}")
    print(f"   Mean Latency:   {df['latency_sec'].mean():.2f} sec/doc")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--models", nargs="+",
                        default=["tesseract", "paddleocr", "qwen2vl"],
                        choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of docs (for smoke testing)")
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset} test split...")
    samples = DATASET_LOADERS[args.dataset]()
    if args.limit:
        samples = samples[:args.limit]
    print(f"✅ Loaded {len(samples)} samples\n")

    # Run each requested model
    all_dfs = {}
    for model_name in args.models:
        print(f"\n{'═'*70}\n{model_name.upper()} on {args.dataset}\n{'═'*70}")
        out_dir = EXTRACTIONS_ROOT / args.dataset / model_name
        results = MODEL_RUNNERS[model_name](samples, args.dataset, out_dir)
        if results:
            df = _summary(results, model_name)
            csv_path = METRICS_DIR / f"{args.dataset.lower()}_{model_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"   Saved: {csv_path}")
            all_dfs[model_name] = df

    # Cross-model comparison
    if len(all_dfs) > 1:
        print(f"\n{'═'*70}\nCOMPARISON — {args.dataset}\n{'═'*70}")
        rows = []
        for name, df in all_dfs.items():
            rows.append({
                "model": name,
                "documents": len(df),
                "precision": round(df["precision"].mean(), 3),
                "recall": round(df["recall"].mean(), 3),
                "f1": round(df["f1"].mean(), 3),
                "latency_sec": round(df["latency_sec"].mean(), 2),
            })
        comp_df = pd.DataFrame(rows)
        print(comp_df.to_string(index=False))
        comp_path = METRICS_DIR / f"{args.dataset.lower()}_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\n✅ Comparison saved: {comp_path}")


if __name__ == "__main__":
    main()
