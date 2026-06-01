"""
Layer 1 Scoring v2 — Field-Level Metrics

Reads the cached OCR/VLM extractions from Layer 1 runs and adds two new
metrics: field-level F1 and field recall. Combined with the existing
token-overlap F1, this gives three complementary views of model performance.

Why this script exists:
    Token-overlap F1 is misleading on datasets with sparse ground truth
    (e.g. SROIE has only 4 key fields per receipt, but OCR extracts ~200
    tokens). Field-level F1 asks the right question: "did the model
    find the specific things we care about?"

Usage:
    cd /mnt/yasir_drive/E_DATA/ResearchProject
    conda activate edata
    python code/layer1_vision/score_v2.py

Outputs:
    results/metrics/full_comparison_v2.csv         (expanded 9-cell table)
    results/metrics/full_comparison_v2.md          (paper-ready markdown)
    results/metrics/<dataset>_field_details.csv    (per-doc field hits)
"""

import json
import re
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional

import pandas as pd


PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTIONS_ROOT = PROJECT_ROOT / "results" / "extractions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DATASETS = ["FUNSD", "SROIE", "CORD"]
MODELS = ["tesseract", "paddleocr", "qwen2vl"]


# ═════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation except internal, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s.,/-]", " ", s)   # keep alphanumerics and a few separators
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_number(s: str) -> Optional[str]:
    """
    Extract canonical numeric value if string contains one.
    '$24.50', '24,50', '24.5', 'RM 24.50' -> '24.5'
    Returns None if no number found.
    """
    if s is None:
        return None
    s = str(s)
    # Common currency/thousand separator cleanup
    cleaned = re.sub(r"[^\d.,-]", "", s)
    # Try to find a decimal number
    matches = re.findall(r"-?\d[\d,]*\.?\d*", cleaned)
    if not matches:
        return None
    best = max(matches, key=len)
    # European format: '24,50' -> '24.50'
    if "," in best and "." not in best:
        best = best.replace(",", ".")
    else:
        best = best.replace(",", "")
    try:
        f = float(best)
        # Strip trailing zeros: 24.50 -> 24.5
        return f"{f:g}"
    except ValueError:
        return None


def text_matches_in(needle: str, haystack: str) -> bool:
    """Substring match after normalization. Handles short tokens carefully."""
    n = normalize_text(needle)
    h = normalize_text(haystack)
    if not n:
        return False
    # For very short needles, require word boundary to avoid spurious matches
    if len(n) <= 3:
        return bool(re.search(rf"\b{re.escape(n)}\b", h))
    return n in h


def number_matches_in(needle: str, haystack: str) -> bool:
    """
    Find the canonical numeric form of needle in haystack.
    Generous: '24.50' matches '$24.50', '24,50', '24.5', etc.
    """
    target = normalize_number(needle)
    if target is None:
        return False
    # Find all numbers in haystack and canonicalize each
    candidates = re.findall(r"-?\d[\d.,]*", haystack)
    for c in candidates:
        if normalize_number(c) == target:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH LOADERS — extract structured fields per dataset
# ═════════════════════════════════════════════════════════════════════════════

def load_sroie_fields(doc_id: str) -> Dict[str, str]:
    """SROIE ground truth: 4 fixed fields per receipt."""
    ann_path = DATA_ROOT / "SROIE2019" / "test" / "entities" / f"{doc_id}.txt"
    if not ann_path.exists():
        return {}
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: str(v) for k, v in data.items() if v}
    except (json.JSONDecodeError, OSError):
        return {}


def load_cord_fields(doc_id: str) -> Dict[str, list]:
    """
    CORD ground truth: extract menu items + totals.
    Returns dict with:
      - 'menu_items': list of menu name strings
      - 'menu_prices': list of menu price strings
      - 'totals':      list of total/subtotal/tax strings
    """
    ann_path = DATA_ROOT / "CORD" / "CORD" / "test" / "json" / f"{doc_id}.json"
    if not ann_path.exists():
        return {"menu_items": [], "menu_prices": [], "totals": []}

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    menu_items, menu_prices, totals = [], [], []
    for line in data.get("valid_line", []):
        category = line.get("category", "")
        for word in line.get("words", []):
            text = word.get("text", "").strip()
            if not text:
                continue
            if category.startswith("menu.nm"):
                menu_items.append(text)
            elif category.startswith("menu.price") or category.startswith("menu.unitprice"):
                menu_prices.append(text)
            elif category.startswith("total.") or category.startswith("sub_total."):
                totals.append(text)

    return {
        "menu_items": menu_items,
        "menu_prices": menu_prices,
        "totals": totals,
    }


def load_funsd_answer_tokens(doc_id: str) -> Set[str]:
    """
    For FUNSD we don't have fixed fields, but we have items labeled 'answer'.
    Return tokens that appear in 'answer'-labeled items only — this is the
    information an SME would actually want to extract.
    """
    ann_path = DATA_ROOT / "dataset" / "testing_data" / "annotations" / f"{doc_id}.json"
    if not ann_path.exists():
        return set()
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = set()
    for item in data.get("form", []):
        if item.get("label") == "answer":
            for tok in str(item.get("text", "")).lower().split():
                cleaned = "".join(c for c in tok if c.isalnum())
                if len(cleaned) > 1:
                    tokens.add(cleaned)
    return tokens


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION LOADERS — read what each model produced
# ═════════════════════════════════════════════════════════════════════════════

def load_extraction(dataset: str, model: str, doc_id: str) -> Optional[Dict]:
    """Read the cached per-doc JSON produced by run_layer1.py."""
    path = EXTRACTIONS_ROOT / dataset / model / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# FIELD-LEVEL SCORING PER DATASET
# ═════════════════════════════════════════════════════════════════════════════

def score_sroie_doc(extracted_text: str, gt_fields: Dict[str, str]) -> Dict:
    """
    For each of SROIE's 4 fields, check if the ground-truth value appears
    in the extracted text (with appropriate matching).
    """
    hits = {}
    for field, gt_value in gt_fields.items():
        if not gt_value:
            hits[field] = None  # field not in ground truth
            continue
        if field == "total":
            # Numeric match
            hits[field] = number_matches_in(gt_value, extracted_text)
        else:
            # Text match (company, date, address)
            hits[field] = text_matches_in(gt_value, extracted_text)

    found = sum(1 for v in hits.values() if v is True)
    total = sum(1 for v in hits.values() if v is not None)
    precision = found / total if total else 0.0   # P = R for fixed-field eval
    return {
        "fields_checked": total,
        "fields_found": found,
        "field_recall": round(precision, 3),
        "hits": hits,
    }


def score_cord_doc(extracted_text: str, gt_fields: Dict[str, list]) -> Dict:
    """
    For CORD, score recall on menu items, menu prices, and totals separately.
    """
    text = extracted_text  # already raw text from OCR/VLM

    def recall_for(values: list, numeric: bool) -> Tuple[int, int]:
        if not values:
            return 0, 0
        hits = 0
        for v in values:
            v = str(v).strip()
            if not v:
                continue
            if numeric:
                if number_matches_in(v, text):
                    hits += 1
            else:
                if text_matches_in(v, text):
                    hits += 1
        return hits, len([v for v in values if str(v).strip()])

    menu_hits, menu_total = recall_for(gt_fields["menu_items"], numeric=False)
    price_hits, price_total = recall_for(gt_fields["menu_prices"], numeric=True)
    total_hits, total_total = recall_for(gt_fields["totals"], numeric=True)

    all_hits = menu_hits + price_hits + total_hits
    all_total = menu_total + price_total + total_total

    return {
        "menu_items_recall": round(menu_hits / menu_total, 3) if menu_total else None,
        "menu_prices_recall": round(price_hits / price_total, 3) if price_total else None,
        "totals_recall": round(total_hits / total_total, 3) if total_total else None,
        "overall_recall": round(all_hits / all_total, 3) if all_total else 0.0,
        "items_checked": all_total,
        "items_found": all_hits,
    }


def score_funsd_doc(extracted_text: str, gt_answer_tokens: Set[str]) -> Dict:
    """
    For FUNSD: how many of the 'answer'-labeled tokens were captured?
    """
    if not gt_answer_tokens:
        return {"answer_recall": None, "answers_checked": 0, "answers_found": 0}

    text_lower = extracted_text.lower()
    tokens_in_text = set()
    for tok in text_lower.split():
        cleaned = "".join(c for c in tok if c.isalnum())
        if len(cleaned) > 1:
            tokens_in_text.add(cleaned)

    found = len(gt_answer_tokens & tokens_in_text)
    total = len(gt_answer_tokens)
    return {
        "answer_recall": round(found / total, 3) if total else 0.0,
        "answers_checked": total,
        "answers_found": found,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — score everything, aggregate, report
# ═════════════════════════════════════════════════════════════════════════════

def score_dataset(dataset: str, model: str) -> Optional[pd.DataFrame]:
    """Score one (dataset, model) combination. Returns per-doc DataFrame."""
    ext_dir = EXTRACTIONS_ROOT / dataset / model
    if not ext_dir.exists():
        return None

    rows = []
    for ext_file in sorted(ext_dir.glob("*.json")):
        doc_id = ext_file.stem
        with open(ext_file, "r", encoding="utf-8") as f:
            ext = json.load(f)

        extracted_text = ext.get("raw_text", "")
        existing_metrics = ext.get("metrics", {})

        row = {
            "doc_id": doc_id,
            "token_precision": existing_metrics.get("precision", 0.0),
            "token_recall":    existing_metrics.get("recall", 0.0),
            "token_f1":        existing_metrics.get("f1", 0.0),
            "latency_sec":     ext.get("latency_sec", 0.0),
        }

        if dataset == "SROIE":
            gt = load_sroie_fields(doc_id)
            r = score_sroie_doc(extracted_text, gt)
            row["field_recall"] = r["field_recall"]
            row["fields_found"] = r["fields_found"]
            row["fields_checked"] = r["fields_checked"]
            for fname, hit in r["hits"].items():
                row[f"hit_{fname}"] = hit

        elif dataset == "CORD":
            gt = load_cord_fields(doc_id)
            r = score_cord_doc(extracted_text, gt)
            row["field_recall"] = r["overall_recall"]
            row["menu_items_recall"] = r["menu_items_recall"]
            row["menu_prices_recall"] = r["menu_prices_recall"]
            row["totals_recall"] = r["totals_recall"]
            row["items_checked"] = r["items_checked"]
            row["items_found"] = r["items_found"]

        elif dataset == "FUNSD":
            gt = load_funsd_answer_tokens(doc_id)
            r = score_funsd_doc(extracted_text, gt)
            row["answer_recall"] = r["answer_recall"]
            row["answers_found"] = r["answers_found"]
            row["answers_checked"] = r["answers_checked"]

        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 78)
    print(" " * 22 + "LAYER 1 SCORING v2 — Field-Level Metrics")
    print("═" * 78)

    summary_rows = []

    for dataset in DATASETS:
        print(f"\n── {dataset} ──")
        for model in MODELS:
            df = score_dataset(dataset, model)
            if df is None:
                print(f"  {model:<12}  no extractions found, skipping")
                continue

            # Save per-doc details
            details_path = METRICS_DIR / f"{dataset.lower()}_{model}_v2.csv"
            df.to_csv(details_path, index=False)

            # Aggregate row
            row = {
                "dataset": dataset,
                "model": model,
                "documents": len(df),
                "token_f1":     round(df["token_f1"].mean(), 3),
                "token_recall": round(df["token_recall"].mean(), 3),
                "latency_sec":  round(df["latency_sec"].mean(), 2),
            }

            if dataset == "SROIE":
                row["field_recall"] = round(df["field_recall"].mean(), 3)
                # Per-field hit rates
                for fname in ["company", "date", "address", "total"]:
                    col = f"hit_{fname}"
                    if col in df.columns:
                        hits = df[col].dropna()
                        row[f"hit_{fname}"] = round(hits.mean(), 3) if len(hits) else None

            elif dataset == "CORD":
                row["field_recall"] = round(df["field_recall"].mean(), 3)
                row["menu_items_recall"]  = round(df["menu_items_recall"].dropna().mean(), 3)
                row["menu_prices_recall"] = round(df["menu_prices_recall"].dropna().mean(), 3)
                row["totals_recall"]      = round(df["totals_recall"].dropna().mean(), 3)

            elif dataset == "FUNSD":
                row["answer_recall"] = round(df["answer_recall"].dropna().mean(), 3)

            summary_rows.append(row)
            print(f"  {model:<12}  docs={len(df):>3}  "
                  f"token_F1={row['token_f1']:.3f}  "
                  + (f"field_recall={row.get('field_recall', 0):.3f}  "
                     if dataset != "FUNSD"
                     else f"answer_recall={row.get('answer_recall', 0):.3f}  ")
                  + f"latency={row['latency_sec']:.2f}s")

    # ─── Build and save the master comparison table ─────────────────────
    if not summary_rows:
        print("\n❌ No results found. Did Layer 1 runs complete?")
        return

    summary_df = pd.DataFrame(summary_rows)
    csv_path = METRICS_DIR / "full_comparison_v2.csv"
    summary_df.to_csv(csv_path, index=False)

    # ─── Print the headline comparison ──────────────────────────────────
    print("\n" + "═" * 78)
    print(" " * 22 + "HEADLINE COMPARISON  (paper Table 1)")
    print("═" * 78)
    print(f"{'Dataset':<8} {'Model':<11} {'Docs':>5} {'TokenF1':>9} "
          f"{'KeyRecall':>10} {'Latency':>9}")
    print("─" * 78)

    last_ds = None
    for r in summary_rows:
        if r["dataset"] != last_ds:
            if last_ds is not None:
                print("·" * 78)
            last_ds = r["dataset"]
        if r["dataset"] == "FUNSD":
            key_metric = r.get("answer_recall", 0)
            key_label = "AnsRecall"
        else:
            key_metric = r.get("field_recall", 0)
            key_label = "FldRecall"
        print(f"{r['dataset']:<8} {r['model']:<11} {r['documents']:>5} "
              f"{r['token_f1']:>9.3f} {key_metric:>10.3f} "
              f"{r['latency_sec']:>8.2f}s")
    print("═" * 78)

    # Per-field breakdown for SROIE
    print("\n── SROIE per-field hit rates ──")
    print(f"{'Model':<12} {'Company':>9} {'Date':>9} {'Address':>9} {'Total':>9}")
    print("─" * 50)
    for r in summary_rows:
        if r["dataset"] == "SROIE":
            print(f"{r['model']:<12} {r.get('hit_company', 0):>9.3f} "
                  f"{r.get('hit_date', 0):>9.3f} {r.get('hit_address', 0):>9.3f} "
                  f"{r.get('hit_total', 0):>9.3f}")

    # Per-field breakdown for CORD
    print("\n── CORD per-field recall ──")
    print(f"{'Model':<12} {'MenuItems':>10} {'Prices':>10} {'Totals':>10}")
    print("─" * 46)
    for r in summary_rows:
        if r["dataset"] == "CORD":
            print(f"{r['model']:<12} "
                  f"{r.get('menu_items_recall', 0):>10.3f} "
                  f"{r.get('menu_prices_recall', 0):>10.3f} "
                  f"{r.get('totals_recall', 0):>10.3f}")

    # ─── Mean by model across datasets (key metric only) ────────────────
    print("\n── Mean Key Recall across datasets (by model) ──")
    by_model_key = {}
    for r in summary_rows:
        model = r["model"]
        key = r.get("answer_recall") if r["dataset"] == "FUNSD" else r.get("field_recall")
        by_model_key.setdefault(model, []).append(key)
    for model in MODELS:
        vals = by_model_key.get(model, [])
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {model:<12} mean key recall = {mean:.3f}")

    # ─── Write markdown report ──────────────────────────────────────────
    md_lines = ["# Layer 1 — Field-Level Comparison (v2)\n",
                "## Headline table\n",
                "| Dataset | Model | Docs | Token F1 | Key Recall | Latency (s) |",
                "|---------|-------|------|----------|-----------|------|"]
    for r in summary_rows:
        key_metric = r.get("answer_recall", 0) if r["dataset"] == "FUNSD" else r.get("field_recall", 0)
        md_lines.append(f"| {r['dataset']} | {r['model']} | {r['documents']} | "
                        f"{r['token_f1']:.3f} | {key_metric:.3f} | {r['latency_sec']:.2f} |")
    md_lines.append("")
    md_lines.append("**Token F1**: token overlap with entire ground-truth annotation (Day 1 metric).")
    md_lines.append("**Key Recall**: SROIE/CORD: how many key fields/items were captured. "
                    "FUNSD: how many tokens marked as 'answer' were captured.")
    md_lines.append("")

    md_lines.append("## SROIE per-field hit rates\n")
    md_lines.append("| Model | Company | Date | Address | Total |")
    md_lines.append("|-------|---------|------|---------|-------|")
    for r in summary_rows:
        if r["dataset"] == "SROIE":
            md_lines.append(f"| {r['model']} | {r.get('hit_company', 0):.3f} | "
                            f"{r.get('hit_date', 0):.3f} | {r.get('hit_address', 0):.3f} | "
                            f"{r.get('hit_total', 0):.3f} |")

    md_lines.append("\n## CORD per-field recall\n")
    md_lines.append("| Model | Menu Items | Menu Prices | Totals |")
    md_lines.append("|-------|-----------|-------------|--------|")
    for r in summary_rows:
        if r["dataset"] == "CORD":
            md_lines.append(f"| {r['model']} | {r.get('menu_items_recall', 0):.3f} | "
                            f"{r.get('menu_prices_recall', 0):.3f} | "
                            f"{r.get('totals_recall', 0):.3f} |")

    md_path = METRICS_DIR / "full_comparison_v2.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n✅ CSV:        {csv_path}")
    print(f"✅ Markdown:   {md_path}")
    print(f"✅ Per-doc CSVs: {METRICS_DIR}/<dataset>_<model>_v2.csv")


if __name__ == "__main__":
    main()
