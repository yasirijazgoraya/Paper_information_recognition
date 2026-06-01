"""
Layer 1 Scoring v3 — Defensible Field-Level Metrics

Improvements over v2:
  • Fix A — CORD numeric matching handles Indonesian thousands-separator
    convention ("25.000" = 25000, not 25.0).
  • Fix B — SROIE address matching uses fuzzy token-set overlap (≥70%)
    instead of strict substring, matching SROIE competition standards.
  • Fix C — FUNSD adds question→answer pairing scoring: did the model
    preserve question-answer relationships, not just capture tokens?
  • Plus: per-field doc-level logging so we can audit failures.

Reads cached extractions from results/extractions/ — no model re-runs.

Usage:
    python code/layer1_vision/score_v3.py

Outputs:
    results/metrics/full_comparison_v3.csv
    results/metrics/full_comparison_v3.md
    results/metrics/<dataset>_<model>_v3.csv         (per-doc detail)
    results/metrics/<dataset>_<model>_failures.csv   (per-doc failures)
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

# Threshold for fuzzy address matching (Fix B)
ADDRESS_TOKEN_OVERLAP_THRESHOLD = 0.70

# Window for FUNSD question-answer proximity (Fix C)
# How many characters around the question text count as 'nearby'
FUNSD_PROXIMITY_CHARS = 200


# ═════════════════════════════════════════════════════════════════════════════
# TEXT NORMALISATION & MATCHING PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation (keep numerics & basic separators), collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s.,/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_number_robust(s: str) -> Optional[str]:
    """
    Canonicalise a numeric string, handling Indonesian thousands-separator (Fix A).

    Examples (input → output):
        '$24.50'      → '24.5'        (US/UK format)
        '24,50'       → '24.5'        (European decimal)
        '25.000'      → '25000'       (Indonesian thousands)
        '1.234.567'   → '1234567'     (Indonesian, multiple groups)
        '1,234.50'    → '1234.5'      (US thousands + decimal)
        'RM 24.50'    → '24.5'        (currency prefix)
        '25.50'       → '25.5'        (true decimal: 2 digits after dot)
    """
    if s is None:
        return None
    raw = str(s)

    # Strip everything except digits, separators, and signs
    cleaned = re.sub(r"[^\d.,-]", "", raw)
    if not cleaned:
        return None

    # Find the largest numeric run
    matches = re.findall(r"-?[\d.,]+", cleaned)
    if not matches:
        return None
    best = max(matches, key=len).strip(".,")
    if not best:
        return None

    # Decide format heuristically
    has_dot = "." in best
    has_comma = "," in best

    if has_dot and has_comma:
        # Both present: last separator is decimal, others are thousands
        if best.rfind(",") > best.rfind("."):
            # European: '1.234,50' → decimal is comma
            best = best.replace(".", "").replace(",", ".")
        else:
            # US: '1,234.50' → decimal is dot
            best = best.replace(",", "")
    elif has_dot:
        # Dot only — could be decimal OR Indonesian thousands
        parts = best.split(".")
        # Indonesian thousands: dot followed by exactly 3 digits, and either
        # multiple dot groups or no other interpretation makes sense
        if all(len(p) == 3 for p in parts[1:]) and len(parts) >= 2 and len(parts[0]) <= 3:
            # e.g. '25.000' or '1.234.567' → thousands
            best = best.replace(".", "")
        # else: keep dot as decimal (e.g. '25.50', '0.99')
    elif has_comma:
        # Comma only — European decimal: '24,50' → '24.50'
        # Unless it looks like thousands ('1,234')
        parts = best.split(",")
        if all(len(p) == 3 for p in parts[1:]) and len(parts) >= 2 and len(parts[0]) <= 3:
            best = best.replace(",", "")
        else:
            best = best.replace(",", ".")

    try:
        f = float(best)
        return f"{f:g}"  # canonical short form, strips trailing zeros
    except ValueError:
        return None


def text_matches_substring(needle: str, haystack: str) -> bool:
    """Strict substring match after normalization (for short or exact text)."""
    n = normalize_text(needle)
    h = normalize_text(haystack)
    if not n:
        return False
    if len(n) <= 3:
        return bool(re.search(rf"\b{re.escape(n)}\b", h))
    return n in h


def text_matches_fuzzy(needle: str, haystack: str, threshold: float = 0.70) -> bool:
    """
    Fuzzy token-set match (Fix B). Returns True if ≥ threshold of the
    significant tokens in `needle` appear in `haystack`.

    Designed for addresses, multi-line names, anything where OCR may shuffle
    punctuation or split lines but the content is mostly there.
    """
    needle_tokens = {t for t in normalize_text(needle).split() if len(t) >= 2}
    haystack_tokens = {t for t in normalize_text(haystack).split() if len(t) >= 2}
    if not needle_tokens:
        return False
    overlap = len(needle_tokens & haystack_tokens) / len(needle_tokens)
    return overlap >= threshold


def number_matches_robust(needle: str, haystack: str) -> bool:
    """
    Find a canonical match for `needle`'s numeric value in `haystack`.
    Uses normalize_number_robust on both sides — handles all major formats.
    """
    target = normalize_number_robust(needle)
    if target is None:
        return False
    candidates = re.findall(r"-?[\d.,]+", haystack)
    for c in candidates:
        if normalize_number_robust(c) == target:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def load_sroie_fields(doc_id: str) -> Dict[str, str]:
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

    return {"menu_items": menu_items, "menu_prices": menu_prices, "totals": totals}


def load_funsd_structured(doc_id: str) -> Dict:
    """
    Load FUNSD with the full structure: items, labels, and linking IDs.
    Returns:
        items:       list of {id, text, label, linking}
        qa_pairs:    list of (question_text, answer_text) — linked pairs
        answer_tokens: set of tokens from answer-labelled items
    """
    ann_path = DATA_ROOT / "dataset" / "testing_data" / "annotations" / f"{doc_id}.json"
    if not ann_path.exists():
        return {"items": [], "qa_pairs": [], "answer_tokens": set()}

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("form", [])
    by_id = {item["id"]: item for item in items if "id" in item}

    # Build linked QA pairs
    qa_pairs = []
    for item in items:
        if item.get("label") == "question":
            q_text = item.get("text", "").strip()
            if not q_text:
                continue
            for link in item.get("linking", []):
                # link is [from_id, to_id]
                other_id = link[1] if link[0] == item.get("id") else link[0]
                if other_id in by_id and by_id[other_id].get("label") == "answer":
                    a_text = by_id[other_id].get("text", "").strip()
                    if a_text:
                        qa_pairs.append((q_text, a_text))

    # Collect all answer tokens
    answer_tokens = set()
    for item in items:
        if item.get("label") == "answer":
            for tok in str(item.get("text", "")).lower().split():
                cleaned = "".join(c for c in tok if c.isalnum())
                if len(cleaned) > 1:
                    answer_tokens.add(cleaned)

    return {"items": items, "qa_pairs": qa_pairs, "answer_tokens": answer_tokens}


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_extraction(dataset: str, model: str, doc_id: str) -> Optional[Dict]:
    path = EXTRACTIONS_ROOT / dataset / model / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# SCORING — SROIE (with Fix B for addresses)
# ═════════════════════════════════════════════════════════════════════════════

def score_sroie_doc(extracted_text: str, gt_fields: Dict[str, str]) -> Dict:
    """For each SROIE field, check ground-truth value in extracted text."""
    hits = {}
    for field, gt_value in gt_fields.items():
        if not gt_value:
            hits[field] = None
            continue

        if field == "total":
            hits[field] = number_matches_robust(gt_value, extracted_text)
        elif field == "address":
            # Fix B: fuzzy token-set match for multi-line addresses
            hits[field] = text_matches_fuzzy(gt_value, extracted_text,
                                              ADDRESS_TOKEN_OVERLAP_THRESHOLD)
        else:
            # company, date — substring is appropriate
            hits[field] = text_matches_substring(gt_value, extracted_text)

    found = sum(1 for v in hits.values() if v is True)
    total = sum(1 for v in hits.values() if v is not None)
    return {
        "fields_checked": total,
        "fields_found": found,
        "field_recall": round(found / total, 3) if total else 0.0,
        "hits": hits,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCORING — CORD (with Fix A for Indonesian numbers)
# ═════════════════════════════════════════════════════════════════════════════

def score_cord_doc(extracted_text: str, gt_fields: Dict[str, list]) -> Dict:
    def recall_for(values: list, numeric: bool) -> Tuple[int, int, list]:
        hits, total, missed = 0, 0, []
        for v in values:
            v = str(v).strip()
            if not v:
                continue
            total += 1
            if numeric:
                ok = number_matches_robust(v, extracted_text)
            else:
                ok = text_matches_substring(v, extracted_text)
            if ok:
                hits += 1
            else:
                missed.append(v)
        return hits, total, missed

    menu_hits, menu_total, menu_missed = recall_for(gt_fields["menu_items"], numeric=False)
    price_hits, price_total, price_missed = recall_for(gt_fields["menu_prices"], numeric=True)
    total_hits, total_total, total_missed = recall_for(gt_fields["totals"], numeric=True)

    all_hits = menu_hits + price_hits + total_hits
    all_total = menu_total + price_total + total_total

    return {
        "menu_items_recall":  round(menu_hits / menu_total, 3) if menu_total else None,
        "menu_prices_recall": round(price_hits / price_total, 3) if price_total else None,
        "totals_recall":      round(total_hits / total_total, 3) if total_total else None,
        "overall_recall":     round(all_hits / all_total, 3) if all_total else 0.0,
        "items_checked":      all_total,
        "items_found":        all_hits,
        "missed_menu_items":  menu_missed[:5],   # capped for storage
        "missed_prices":      price_missed[:5],
        "missed_totals":      total_missed[:5],
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCORING — FUNSD (with Fix C: question-answer pairing)
# ═════════════════════════════════════════════════════════════════════════════

def _token_set(text: str) -> Set[str]:
    return {
        "".join(c for c in tok if c.isalnum())
        for tok in normalize_text(text).split()
        if len("".join(c for c in tok if c.isalnum())) > 1
    }


def score_funsd_doc(extracted_text: str, gt_structure: Dict) -> Dict:
    """
    Two FUNSD metrics:
      (a) answer_recall  — fraction of all answer-labelled tokens captured (v2 metric)
      (b) qa_pair_recall — for each linked Q->A pair, did the model capture
                           the answer text within proximity of the question text?
    """
    extracted_lower = extracted_text.lower()
    extracted_tokens = _token_set(extracted_text)

    # (a) answer-token recall
    answer_tokens = gt_structure["answer_tokens"]
    if answer_tokens:
        ans_recall = round(len(answer_tokens & extracted_tokens) / len(answer_tokens), 3)
    else:
        ans_recall = None

    # (c) Q-A pair recall (Fix C)
    qa_pairs = gt_structure["qa_pairs"]
    pair_hits = 0
    pair_failed = []
    for q_text, a_text in qa_pairs:
        q_norm = normalize_text(q_text)
        a_norm = normalize_text(a_text)
        if not q_norm or not a_norm:
            continue

        # Look for question text in extraction (any occurrence)
        q_pos = extracted_lower.find(q_norm[:30]) if len(q_norm) >= 5 else -1
        if q_pos == -1:
            # Question not found at all — soft fail
            pair_failed.append({"q": q_text, "a": a_text, "reason": "question not found"})
            continue

        # Look for answer text in proximity to the question
        window_start = max(0, q_pos - FUNSD_PROXIMITY_CHARS)
        window_end = min(len(extracted_lower),
                         q_pos + len(q_norm) + FUNSD_PROXIMITY_CHARS)
        window = extracted_lower[window_start:window_end]

        # Match: substantial tokens of the answer appear in the window
        a_tokens = _token_set(a_text)
        if not a_tokens:
            continue
        window_tokens = _token_set(window)
        overlap = len(a_tokens & window_tokens) / len(a_tokens)
        if overlap >= 0.5:
            pair_hits += 1
        else:
            pair_failed.append({"q": q_text, "a": a_text, "reason": f"answer overlap {overlap:.2f}"})

    pair_total = len(qa_pairs)
    qa_recall = round(pair_hits / pair_total, 3) if pair_total else None

    return {
        "answer_recall": ans_recall,
        "answers_checked": len(answer_tokens),
        "answers_found": len(answer_tokens & extracted_tokens) if answer_tokens else 0,
        "qa_pair_recall": qa_recall,
        "qa_pairs_total": pair_total,
        "qa_pairs_hit": pair_hits,
        "qa_failures_sample": pair_failed[:5],   # cap for storage
    }


# ═════════════════════════════════════════════════════════════════════════════
# DRIVERS
# ═════════════════════════════════════════════════════════════════════════════

def score_dataset(dataset: str, model: str) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    """Score one (dataset, model) combination. Returns (per-doc DataFrame, failures list)."""
    ext_dir = EXTRACTIONS_ROOT / dataset / model
    if not ext_dir.exists():
        return None, []

    rows = []
    failures = []
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
            # Log failures
            for fname, hit in r["hits"].items():
                if hit is False:
                    failures.append({
                        "doc_id": doc_id, "field": fname,
                        "expected": gt.get(fname, ""),
                    })

        elif dataset == "CORD":
            gt = load_cord_fields(doc_id)
            r = score_cord_doc(extracted_text, gt)
            row["field_recall"] = r["overall_recall"]
            row["menu_items_recall"]  = r["menu_items_recall"]
            row["menu_prices_recall"] = r["menu_prices_recall"]
            row["totals_recall"]      = r["totals_recall"]
            row["items_checked"]      = r["items_checked"]
            row["items_found"]        = r["items_found"]
            for cat, missed in [("menu_items", r["missed_menu_items"]),
                                 ("prices", r["missed_prices"]),
                                 ("totals", r["missed_totals"])]:
                for m in missed:
                    failures.append({"doc_id": doc_id, "field": cat, "expected": m})

        elif dataset == "FUNSD":
            gt = load_funsd_structured(doc_id)
            r = score_funsd_doc(extracted_text, gt)
            row["answer_recall"]   = r["answer_recall"]
            row["qa_pair_recall"]  = r["qa_pair_recall"]
            row["qa_pairs_total"]  = r["qa_pairs_total"]
            row["qa_pairs_hit"]    = r["qa_pairs_hit"]
            for f in r["qa_failures_sample"]:
                failures.append({
                    "doc_id": doc_id, "field": "qa_pair",
                    "expected": f"Q: {f['q']} → A: {f['a']}",
                    "reason": f["reason"],
                })

        rows.append(row)

    return pd.DataFrame(rows) if rows else None, failures


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 78)
    print(" " * 18 + "LAYER 1 SCORING v3 — Defensible Field-Level Metrics")
    print("═" * 78)

    summary_rows = []

    for dataset in DATASETS:
        print(f"\n── {dataset} ──")
        for model in MODELS:
            df, failures = score_dataset(dataset, model)
            if df is None:
                print(f"  {model:<12}  no extractions found, skipping")
                continue

            df.to_csv(METRICS_DIR / f"{dataset.lower()}_{model}_v3.csv", index=False)
            if failures:
                pd.DataFrame(failures).to_csv(
                    METRICS_DIR / f"{dataset.lower()}_{model}_failures.csv", index=False
                )

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
                for fname in ["company", "date", "address", "total"]:
                    col = f"hit_{fname}"
                    if col in df.columns:
                        hits = df[col].dropna()
                        row[f"hit_{fname}"] = round(hits.mean(), 3) if len(hits) else None
                print(f"  {model:<12}  docs={len(df):>3}  token_F1={row['token_f1']:.3f}  "
                      f"field_recall={row['field_recall']:.3f}  latency={row['latency_sec']:.2f}s")

            elif dataset == "CORD":
                row["field_recall"] = round(df["field_recall"].mean(), 3)
                row["menu_items_recall"]  = round(df["menu_items_recall"].dropna().mean(), 3)
                row["menu_prices_recall"] = round(df["menu_prices_recall"].dropna().mean(), 3)
                row["totals_recall"]      = round(df["totals_recall"].dropna().mean(), 3)
                print(f"  {model:<12}  docs={len(df):>3}  token_F1={row['token_f1']:.3f}  "
                      f"field_recall={row['field_recall']:.3f}  latency={row['latency_sec']:.2f}s")

            elif dataset == "FUNSD":
                row["answer_recall"]  = round(df["answer_recall"].dropna().mean(), 3)
                row["qa_pair_recall"] = round(df["qa_pair_recall"].dropna().mean(), 3)
                print(f"  {model:<12}  docs={len(df):>3}  token_F1={row['token_f1']:.3f}  "
                      f"answer_recall={row['answer_recall']:.3f}  "
                      f"qa_pair_recall={row['qa_pair_recall']:.3f}  "
                      f"latency={row['latency_sec']:.2f}s")

            summary_rows.append(row)

    if not summary_rows:
        print("\n❌ No results found.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(METRICS_DIR / "full_comparison_v3.csv", index=False)

    # ─── Console headline ───────────────────────────────────────────────
    print("\n" + "═" * 78)
    print(" " * 24 + "HEADLINE COMPARISON v3")
    print("═" * 78)
    print(f"{'Dataset':<8} {'Model':<11} {'Docs':>5} {'TokenF1':>9} "
          f"{'KeyRecall':>10} {'Strict':>8} {'Latency':>9}")
    print("─" * 78)
    last_ds = None
    for r in summary_rows:
        if r["dataset"] != last_ds:
            if last_ds is not None:
                print("·" * 78)
            last_ds = r["dataset"]
        if r["dataset"] == "FUNSD":
            key_metric = r.get("answer_recall", 0)
            strict_metric = r.get("qa_pair_recall", 0)
        else:
            key_metric = r.get("field_recall", 0)
            strict_metric = float("nan")  # only FUNSD has strict variant
        strict_str = f"{strict_metric:>8.3f}" if not pd.isna(strict_metric) else f"{'—':>8}"
        print(f"{r['dataset']:<8} {r['model']:<11} {r['documents']:>5} "
              f"{r['token_f1']:>9.3f} {key_metric:>10.3f} {strict_str} "
              f"{r['latency_sec']:>8.2f}s")
    print("═" * 78)

    # SROIE per-field
    print("\n── SROIE per-field hit rates (Fix B applied to address) ──")
    print(f"{'Model':<12} {'Company':>9} {'Date':>9} {'Address':>9} {'Total':>9}")
    print("─" * 52)
    for r in summary_rows:
        if r["dataset"] == "SROIE":
            print(f"{r['model']:<12} {r.get('hit_company', 0):>9.3f} "
                  f"{r.get('hit_date', 0):>9.3f} {r.get('hit_address', 0):>9.3f} "
                  f"{r.get('hit_total', 0):>9.3f}")

    # CORD per-field
    print("\n── CORD per-field recall (Fix A applied to numbers) ──")
    print(f"{'Model':<12} {'MenuItems':>10} {'Prices':>10} {'Totals':>10}")
    print("─" * 46)
    for r in summary_rows:
        if r["dataset"] == "CORD":
            print(f"{r['model']:<12} "
                  f"{r.get('menu_items_recall', 0):>10.3f} "
                  f"{r.get('menu_prices_recall', 0):>10.3f} "
                  f"{r.get('totals_recall', 0):>10.3f}")

    # FUNSD strict
    print("\n── FUNSD: Question-Answer pair recall (Fix C) ──")
    print(f"{'Model':<12} {'AnsTokens':>10} {'Q→A Pair':>10}")
    print("─" * 36)
    for r in summary_rows:
        if r["dataset"] == "FUNSD":
            print(f"{r['model']:<12} {r.get('answer_recall', 0):>10.3f} "
                  f"{r.get('qa_pair_recall', 0):>10.3f}")

    # Aggregate
    print("\n── Mean Key Recall (lenient) across datasets ──")
    by_model = {}
    for r in summary_rows:
        key = r.get("answer_recall") if r["dataset"] == "FUNSD" else r.get("field_recall")
        by_model.setdefault(r["model"], []).append(key)
    for model in MODELS:
        vals = [v for v in by_model.get(model, []) if v is not None]
        if vals:
            print(f"  {model:<12} mean = {sum(vals)/len(vals):.3f}")

    # ─── Markdown report ───────────────────────────────────────────────
    md = ["# Layer 1 — Field-Level Comparison v3\n",
          "## Methodology improvements\n",
          "- **Fix A**: CORD numeric matching handles Indonesian thousands "
          "separator (`25.000` = 25000, not 25.0).",
          "- **Fix B**: SROIE address matching uses 70% token-set overlap "
          "(matches multi-line addresses with minor OCR perturbations).",
          "- **Fix C**: FUNSD adds question→answer pair recall — checks "
          "that question and answer appear together, not just both present.\n",
          "## Headline table\n",
          "| Dataset | Model | Docs | Token F1 | Key Recall | Strict Q→A | Latency (s) |",
          "|---------|-------|------|----------|-----------|-----------|------|"]
    for r in summary_rows:
        if r["dataset"] == "FUNSD":
            key = r.get("answer_recall", 0)
            strict = f"{r.get('qa_pair_recall', 0):.3f}"
        else:
            key = r.get("field_recall", 0)
            strict = "—"
        md.append(f"| {r['dataset']} | {r['model']} | {r['documents']} | "
                  f"{r['token_f1']:.3f} | {key:.3f} | {strict} | "
                  f"{r['latency_sec']:.2f} |")

    md.append("\n## SROIE per-field hit rates (Fix B)\n")
    md.append("| Model | Company | Date | Address | Total |")
    md.append("|-------|---------|------|---------|-------|")
    for r in summary_rows:
        if r["dataset"] == "SROIE":
            md.append(f"| {r['model']} | {r.get('hit_company', 0):.3f} | "
                      f"{r.get('hit_date', 0):.3f} | {r.get('hit_address', 0):.3f} | "
                      f"{r.get('hit_total', 0):.3f} |")

    md.append("\n## CORD per-field recall (Fix A)\n")
    md.append("| Model | Menu Items | Menu Prices | Totals |")
    md.append("|-------|-----------|-------------|--------|")
    for r in summary_rows:
        if r["dataset"] == "CORD":
            md.append(f"| {r['model']} | {r.get('menu_items_recall', 0):.3f} | "
                      f"{r.get('menu_prices_recall', 0):.3f} | "
                      f"{r.get('totals_recall', 0):.3f} |")

    md.append("\n## FUNSD: lenient vs strict (Fix C)\n")
    md.append("| Model | Answer-token Recall | Q→A Pair Recall |")
    md.append("|-------|---------------------|------------------|")
    for r in summary_rows:
        if r["dataset"] == "FUNSD":
            md.append(f"| {r['model']} | {r.get('answer_recall', 0):.3f} | "
                      f"{r.get('qa_pair_recall', 0):.3f} |")

    (METRICS_DIR / "full_comparison_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n✅ CSV: {METRICS_DIR / 'full_comparison_v3.csv'}")
    print(f"✅ MD:  {METRICS_DIR / 'full_comparison_v3.md'}")
    print(f"✅ Per-doc CSVs and failure logs in {METRICS_DIR}")


if __name__ == "__main__":
    main()
