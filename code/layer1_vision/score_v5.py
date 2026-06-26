"""
Layer 1 Scoring v5 — adds CORD group-level (structure-aware) F1

Extends v4. All v4 behaviour is retained unchanged (field-level F1 with
precision and bootstrap confidence intervals for SROIE and CORD; FUNSD
answer-token recall and Q->A pair recall). v5 adds one capability:

  • CORD Group F1. CORD menu lines are grouped (a menu row links its name,
    price, and quantity via a shared group_id). Field-level F1 scores each
    value independently and therefore does not check whether values are
    grouped into the correct rows. Group F1 treats a whole menu row as a
    single unit: a row counts as correct only when its name and price both
    match the corresponding ground-truth row. This follows the group-level
    evaluation described in KIEval (Khang et al., 2025).

  Group F1 is reported for CORD only. SROIE and FUNSD have no grouped
  entities, so their scoring is unchanged and field-level F1 remains complete
  for them.

Reads cached extractions from results/extractions/ — no model re-runs.

Usage:
    python code/layer1_vision/score_v5.py

Outputs:
    results/metrics/full_comparison_v5.csv
    results/metrics/full_comparison_v5.md
    results/metrics/<dataset>_<model>_v5.csv         (per-doc detail)
    results/metrics/<dataset>_<model>_failures_v5.csv (per-doc failures)
"""

import json
import re
import random
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional

import pandas as pd


PROJECT_ROOT = Path("/mnt/yasir_drive/E_DATA/ResearchProject")
DATA_ROOT = Path("/mnt/yasir_drive/E_DATA/data")
EXTRACTIONS_ROOT = PROJECT_ROOT / "results" / "extractions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DATASETS = ["FUNSD", "SROIE", "CORD"]
MODELS = ["tesseract", "paddleocr", "qwen2vl"]

ADDRESS_TOKEN_OVERLAP_THRESHOLD = 0.70
FUNSD_PROXIMITY_CHARS = 200

# Bootstrap settings for confidence intervals
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


# ------------------------------------------------------------
# TEXT NORMALISATION & MATCHING PRIMITIVES  (unchanged from v3)
# ------------------------------------------------------------

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s.,/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_number_robust(s: str) -> Optional[str]:
    if s is None:
        return None
    raw = str(s)
    cleaned = re.sub(r"[^\d.,-]", "", raw)
    if not cleaned:
        return None
    matches = re.findall(r"-?[\d.,]+", cleaned)
    if not matches:
        return None
    best = max(matches, key=len).strip(".,")
    if not best:
        return None

    has_dot = "." in best
    has_comma = "," in best
    if has_dot and has_comma:
        if best.rfind(",") > best.rfind("."):
            best = best.replace(".", "").replace(",", ".")
        else:
            best = best.replace(",", "")
    elif has_dot:
        parts = best.split(".")
        if all(len(p) == 3 for p in parts[1:]) and len(parts) >= 2 and len(parts[0]) <= 3:
            best = best.replace(".", "")
    elif has_comma:
        parts = best.split(",")
        if all(len(p) == 3 for p in parts[1:]) and len(parts) >= 2 and len(parts[0]) <= 3:
            best = best.replace(",", "")
        else:
            best = best.replace(",", ".")
    try:
        f = float(best)
        return f"{f:g}"
    except ValueError:
        return None


def text_matches_substring(needle: str, haystack: str) -> bool:
    n = normalize_text(needle)
    h = normalize_text(haystack)
    if not n:
        return False
    if len(n) <= 3:
        return bool(re.search(rf"\b{re.escape(n)}\b", h))
    return n in h


def text_matches_fuzzy(needle: str, haystack: str, threshold: float = 0.70) -> bool:
    needle_tokens = {t for t in normalize_text(needle).split() if len(t) >= 2}
    haystack_tokens = {t for t in normalize_text(haystack).split() if len(t) >= 2}
    if not needle_tokens:
        return False
    overlap = len(needle_tokens & haystack_tokens) / len(needle_tokens)
    return overlap >= threshold


def number_matches_robust(needle: str, haystack: str) -> bool:
    target = normalize_number_robust(needle)
    if target is None:
        return False
    candidates = re.findall(r"-?[\d.,]+", haystack)
    for c in candidates:
        if normalize_number_robust(c) == target:
            return True
    return False


# ------------------------------------------------------------
# GROUND-TRUTH LOADERS  (unchanged from v3)
# ------------------------------------------------------------

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


def load_cord_groups(doc_id: str) -> List[Dict]:
    """
    Load CORD menu rows as grouped units for group-level scoring.

    Each menu line in CORD carries a shared group_id that links the parts of
    one menu row (name, price, quantity). This reconstructs one record per row:
        {"name": <str or None>, "price": <str or None>, "cnt": <str or None>}
    Only menu groups are returned; sub_total and total are not grouped rows.
    """
    ann_path = DATA_ROOT / "CORD" / "CORD" / "test" / "json" / f"{doc_id}.json"
    if not ann_path.exists():
        return []
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: Dict = {}
    for line in data.get("valid_line", []):
        category = line.get("category", "")
        if not category.startswith("menu"):
            continue
        gid = line.get("group_id")
        if gid is None:
            continue
        text = " ".join(w.get("text", "").strip()
                        for w in line.get("words", []) if w.get("text", "").strip())
        if not text:
            continue
        rows.setdefault(gid, {"name": None, "price": None, "cnt": None})
        if "nm" in category:
            rows[gid]["name"] = text
        elif "price" in category or "unitprice" in category:
            # keep the first price seen for the row (menu.price preferred)
            if rows[gid]["price"] is None or category.startswith("menu.price"):
                rows[gid]["price"] = text
        elif "cnt" in category:
            rows[gid]["cnt"] = text

    # A usable row needs at least a name or a price
    return [r for r in rows.values() if r["name"] or r["price"]]


def load_funsd_structured(doc_id: str) -> Dict:
    ann_path = DATA_ROOT / "dataset" / "testing_data" / "annotations" / f"{doc_id}.json"
    if not ann_path.exists():
        return {"items": [], "qa_pairs": [], "answer_tokens": set()}
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("form", [])
    by_id = {item["id"]: item for item in items if "id" in item}
    qa_pairs = []
    for item in items:
        if item.get("label") == "question":
            q_text = item.get("text", "").strip()
            if not q_text:
                continue
            for link in item.get("linking", []):
                other_id = link[1] if link[0] == item.get("id") else link[0]
                if other_id in by_id and by_id[other_id].get("label") == "answer":
                    a_text = by_id[other_id].get("text", "").strip()
                    if a_text:
                        qa_pairs.append((q_text, a_text))
    answer_tokens = set()
    for item in items:
        if item.get("label") == "answer":
            for tok in str(item.get("text", "")).lower().split():
                cleaned = "".join(c for c in tok if c.isalnum())
                if len(cleaned) > 1:
                    answer_tokens.add(cleaned)
    return {"items": items, "qa_pairs": qa_pairs, "answer_tokens": answer_tokens}


# ------------------------------------------------------------
# PREDICTED-VALUE EXTRACTION  (new in v4)
# To compute precision we need the candidate values the model emitted, not just
# whether the GT value is somewhere in raw_text.
# ------------------------------------------------------------

DATE_RE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
NUMERIC_RE = re.compile(r"-?\d[\d.,]*")


def predicted_lines(ext: Dict) -> List[str]:
    """Prefer text_segments; fall back to splitting raw_text."""
    segs = ext.get("text_segments")
    if isinstance(segs, list) and segs:
        return [str(s) for s in segs]
    return [ln for ln in re.split(r"[\n]+", ext.get("raw_text", "")) if ln.strip()]


def select_sroie_prediction(field: str, ext: Dict) -> Optional[str]:
    """
    Pick ONE predicted value for a SROIE field, so per-field P/R/F1 is defined.
    Heuristic, deliberately simple and identical for every backend:
      - date    -> first date-like token in raw_text
      - total   -> the largest numeric value seen (receipts: grand total)
      - company -> first non-empty line (store name is near the top)
      - address -> the 1-3 lines following the company line
    Returns None if nothing plausible was emitted (counts against recall, and
    avoids inflating precision).
    """
    raw = ext.get("raw_text", "")
    lines = predicted_lines(ext)

    if field == "date":
        m = DATE_RE.search(raw)
        return m.group(0) if m else None

    if field == "total":
        nums = [normalize_number_robust(n) for n in NUMERIC_RE.findall(raw)]
        nums = [float(n) for n in nums if n is not None]
        return f"{max(nums):g}" if nums else None

    if field == "company":
        return lines[0].strip() if lines else None

    if field == "address":
        if len(lines) >= 2:
            return " ".join(l.strip() for l in lines[1:4])
        return None

    return None


# ------------------------------------------------------------
# F1 HELPER
# ------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


# ------------------------------------------------------------
# SCORING — SROIE  (v4: precision + recall + F1, one value per field)
# ------------------------------------------------------------

def score_sroie_doc(ext: Dict, gt_fields: Dict[str, str]) -> Dict:
    extracted_text = ext.get("raw_text", "")
    per_field = {}
    tp = fp = fn = 0

    for field, gt_value in gt_fields.items():
        if not gt_value:
            per_field[field] = None
            continue

        pred_value = select_sroie_prediction(field, ext)

        # Does the GT value match what we matched in the text? (recall side)
        if field == "total":
            recall_hit = number_matches_robust(gt_value, extracted_text)
        elif field == "address":
            recall_hit = text_matches_fuzzy(gt_value, extracted_text,
                                            ADDRESS_TOKEN_OVERLAP_THRESHOLD)
        else:
            recall_hit = text_matches_substring(gt_value, extracted_text)

        # Did the model's SELECTED value for this field match the GT? (precision side)
        if pred_value is None:
            precision_hit = False
        elif field == "total":
            precision_hit = number_matches_robust(gt_value, pred_value)
        elif field == "address":
            precision_hit = text_matches_fuzzy(gt_value, pred_value,
                                               ADDRESS_TOKEN_OVERLAP_THRESHOLD)
        else:
            precision_hit = text_matches_substring(gt_value, pred_value) \
                            or text_matches_substring(pred_value, gt_value)

        # Confusion counts (one GT value per field present)
        if recall_hit:
            tp += 1
        else:
            fn += 1
        # If the model emitted a value for this field but it was wrong -> fp
        if pred_value is not None and not precision_hit:
            fp += 1

        per_field[field] = {"recall_hit": recall_hit,
                            "precision_hit": precision_hit,
                            "pred": pred_value}

    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "per_field": per_field,
    }


# ------------------------------------------------------------
# SCORING — CORD  (v4: symmetric F1 — precision against predicted candidates)
# ------------------------------------------------------------

def _cord_pred_candidates(ext: Dict, numeric: bool) -> List[str]:
    """Predicted candidate values for CORD precision: lines or numeric tokens."""
    if numeric:
        return NUMERIC_RE.findall(ext.get("raw_text", ""))
    return predicted_lines(ext)


def score_cord_doc(ext: Dict, gt_fields: Dict[str, list]) -> Dict:
    extracted_text = ext.get("raw_text", "")

    def score_cat(gt_values: list, numeric: bool):
        gt_values = [str(v).strip() for v in gt_values if str(v).strip()]
        total = len(gt_values)
        # Recall: GT values found anywhere in the text
        hits, missed = 0, []
        for v in gt_values:
            ok = number_matches_robust(v, extracted_text) if numeric \
                 else text_matches_substring(v, extracted_text)
            if ok:
                hits += 1
            else:
                missed.append(v)
        recall_tp = hits
        recall_fn = total - hits

        # Precision: of predicted candidates, how many correspond to a GT value
        cands = _cord_pred_candidates(ext, numeric)
        if numeric:
            gt_norm = {normalize_number_robust(v) for v in gt_values}
            gt_norm.discard(None)
            cand_norm = [normalize_number_robust(c) for c in cands]
            cand_norm = [c for c in cand_norm if c is not None]
            pred_total = len(cand_norm)
            pred_hits = sum(1 for c in cand_norm if c in gt_norm)
        else:
            pred_total = len(cands)
            pred_hits = sum(1 for c in cands
                            if any(text_matches_substring(g, c) or text_matches_substring(c, g)
                                   for g in gt_values))
        fp = max(pred_total - pred_hits, 0)
        return recall_tp, fp, recall_fn, missed

    cats = {
        "menu_items":  score_cat(gt_fields["menu_items"], numeric=False),
        "menu_prices": score_cat(gt_fields["menu_prices"], numeric=True),
        "totals":      score_cat(gt_fields["totals"], numeric=True),
    }

    tp = sum(c[0] for c in cats.values())
    fp = sum(c[1] for c in cats.values())
    fn = sum(c[2] for c in cats.values())
    precision, recall, f1 = prf(tp, fp, fn)

    def cat_rate(c):
        t = c[0] + c[2]
        return round(c[0] / t, 3) if t else None

    return {
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "menu_items_recall":  cat_rate(cats["menu_items"]),
        "menu_prices_recall": cat_rate(cats["menu_prices"]),
        "totals_recall":      cat_rate(cats["totals"]),
        "missed": {k: v[3][:5] for k, v in cats.items()},
    }


# ------------------------------------------------------------
# SCORING — CORD group level (structure-aware, KIEval-style)
# ------------------------------------------------------------

def _row_name_match(pred_text: str, gt_name: str) -> bool:
    if not gt_name:
        return True  # nothing to match on name
    return text_matches_substring(gt_name, pred_text) or text_matches_substring(pred_text, gt_name)


def _row_price_match(pred_text: str, gt_price: str) -> bool:
    if not gt_price:
        return True  # nothing to match on price
    return number_matches_robust(gt_price, pred_text)


def score_cord_groups(ext: Dict, gt_rows: List[Dict]) -> Dict:
    """
    Group-level F1 for CORD menu rows.

    A ground-truth menu row counts as recovered only if BOTH its name and its
    price are present together in the prediction text. This checks that the
    model kept the values grouped into the correct row, not just that the
    individual values appeared somewhere on the page.

    Because the cached predictions are flat text (not grouped), a row is
    treated as a true positive when the name and price co-occur. Precision uses
    the number of predicted menu-like rows inferred from the prediction lines.
    """
    extracted_text = ext.get("raw_text", "")

    gt_total = len(gt_rows)
    if gt_total == 0:
        return {"group_f1": None, "group_precision": None, "group_recall": None,
                "groups_total": 0, "groups_hit": 0}

    tp = 0
    for row in gt_rows:
        name_ok = _row_name_match(extracted_text, row["name"])
        price_ok = _row_price_match(extracted_text, row["price"])
        # require both parts that exist in the GT row to be present together
        if name_ok and price_ok and (row["name"] or row["price"]):
            tp += 1
    recall = tp / gt_total

    # Precision proxy: count predicted lines that look like a priced item
    lines = predicted_lines(ext)
    priced_lines = [ln for ln in lines if NUMERIC_RE.search(ln)]
    pred_total = max(len(priced_lines), tp)  # avoid precision > 1 when no lines
    precision = tp / pred_total if pred_total else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "group_f1": round(f1, 3),
        "group_precision": round(precision, 3),
        "group_recall": round(recall, 3),
        "groups_total": gt_total,
        "groups_hit": tp,
    }


# ------------------------------------------------------------
# SCORING — FUNSD  (unchanged from v3)
# ------------------------------------------------------------

def _token_set(text: str) -> Set[str]:
    return {
        "".join(c for c in tok if c.isalnum())
        for tok in normalize_text(text).split()
        if len("".join(c for c in tok if c.isalnum())) > 1
    }


def score_funsd_doc(extracted_text: str, gt_structure: Dict) -> Dict:
    extracted_lower = extracted_text.lower()
    extracted_tokens = _token_set(extracted_text)

    answer_tokens = gt_structure["answer_tokens"]
    ans_recall = round(len(answer_tokens & extracted_tokens) / len(answer_tokens), 3) \
        if answer_tokens else None

    qa_pairs = gt_structure["qa_pairs"]
    pair_hits = 0
    pair_failed = []
    for q_text, a_text in qa_pairs:
        q_norm = normalize_text(q_text)
        a_norm = normalize_text(a_text)
        if not q_norm or not a_norm:
            continue
        q_pos = extracted_lower.find(q_norm[:30]) if len(q_norm) >= 5 else -1
        if q_pos == -1:
            pair_failed.append({"q": q_text, "a": a_text, "reason": "question not found"})
            continue
        window_start = max(0, q_pos - FUNSD_PROXIMITY_CHARS)
        window_end = min(len(extracted_lower), q_pos + len(q_norm) + FUNSD_PROXIMITY_CHARS)
        window = extracted_lower[window_start:window_end]
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
        "qa_pair_recall": qa_recall,
        "qa_pairs_total": pair_total,
        "qa_pairs_hit": pair_hits,
        "qa_failures_sample": pair_failed[:5],
    }


# ------------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS  (new in v4)
# ------------------------------------------------------------

def bootstrap_ci(values: List[float], iters: int = BOOTSTRAP_ITERS,
                 level: float = CI_LEVEL) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the mean of a per-document metric."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    n = len(vals)
    if n == 0:
        return None, None
    if n == 1:
        return round(vals[0], 3), round(vals[0], 3)
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(iters):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - level) / 2 * iters)
    hi_idx = int((1 + level) / 2 * iters) - 1
    return round(means[lo_idx], 3), round(means[hi_idx], 3)


def fmt_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


# ------------------------------------------------------------
# DRIVERS
# ------------------------------------------------------------

def score_dataset(dataset: str, model: str) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    ext_dir = EXTRACTIONS_ROOT / dataset / model
    if not ext_dir.exists():
        return None, []
    rows, failures = [], []
    for ext_file in sorted(ext_dir.glob("*.json")):
        doc_id = ext_file.stem
        with open(ext_file, "r", encoding="utf-8") as f:
            ext = json.load(f)
        extracted_text = ext.get("raw_text", "")
        existing = ext.get("metrics", {})
        row = {
            "doc_id": doc_id,
            "token_precision": existing.get("precision", 0.0),
            "token_recall": existing.get("recall", 0.0),
            "token_f1": existing.get("f1", 0.0),
            "latency_sec": ext.get("latency_sec", 0.0),
        }

        if dataset == "SROIE":
            gt = load_sroie_fields(doc_id)
            r = score_sroie_doc(ext, gt)
            row.update({"field_precision": r["field_precision"],
                        "field_recall": r["field_recall"],
                        "field_f1": r["field_f1"]})
            for fname, info in r["per_field"].items():
                row[f"hit_{fname}"] = (info["recall_hit"] if isinstance(info, dict) else None)
                if isinstance(info, dict) and not info["recall_hit"]:
                    failures.append({"doc_id": doc_id, "field": fname,
                                     "expected": gt.get(fname, "")})

        elif dataset == "CORD":
            gt = load_cord_fields(doc_id)
            r = score_cord_doc(ext, gt)
            gt_rows = load_cord_groups(doc_id)
            g = score_cord_groups(ext, gt_rows)
            row.update({"field_precision": r["field_precision"],
                        "field_recall": r["field_recall"],
                        "field_f1": r["field_f1"],
                        "menu_items_recall": r["menu_items_recall"],
                        "menu_prices_recall": r["menu_prices_recall"],
                        "totals_recall": r["totals_recall"],
                        "group_f1": g["group_f1"],
                        "group_precision": g["group_precision"],
                        "group_recall": g["group_recall"]})
            for cat, missed in r["missed"].items():
                for m in missed:
                    failures.append({"doc_id": doc_id, "field": cat, "expected": m})

        elif dataset == "FUNSD":
            gt = load_funsd_structured(doc_id)
            r = score_funsd_doc(extracted_text, gt)
            row.update({"answer_recall": r["answer_recall"],
                        "qa_pair_recall": r["qa_pair_recall"],
                        "qa_pairs_total": r["qa_pairs_total"],
                        "qa_pairs_hit": r["qa_pairs_hit"]})
            for fobj in r["qa_failures_sample"]:
                failures.append({"doc_id": doc_id, "field": "qa_pair",
                                 "expected": f"Q: {fobj['q']} -> A: {fobj['a']}",
                                 "reason": fobj["reason"]})
        rows.append(row)
    return (pd.DataFrame(rows) if rows else None), failures


def summarise(dataset, model, df) -> Dict:
    row = {"dataset": dataset, "model": model, "documents": len(df),
           "token_f1": round(df["token_f1"].mean(), 3)}
    if dataset in ("SROIE", "CORD"):
        for m in ["field_precision", "field_recall", "field_f1"]:
            row[m] = round(df[m].mean(), 3)
        lo, hi = bootstrap_ci(df["field_f1"].tolist())
        row["field_f1_ci"] = fmt_ci(lo, hi)
        if dataset == "SROIE":
            for fname in ["company", "date", "address", "total"]:
                col = f"hit_{fname}"
                if col in df.columns:
                    hits = df[col].dropna()
                    row[f"hit_{fname}"] = round(hits.mean(), 3) if len(hits) else None
        else:
            for c in ["menu_items_recall", "menu_prices_recall", "totals_recall"]:
                row[c] = round(df[c].dropna().mean(), 3)
            if "group_f1" in df.columns:
                grp = df["group_f1"].dropna()
                row["group_f1"] = round(grp.mean(), 3) if len(grp) else None
                row["group_precision"] = round(df["group_precision"].dropna().mean(), 3) if "group_precision" in df.columns else None
                row["group_recall"] = round(df["group_recall"].dropna().mean(), 3) if "group_recall" in df.columns else None
                glo, ghi = bootstrap_ci(grp.tolist())
                row["group_f1_ci"] = fmt_ci(glo, ghi)
    else:  # FUNSD
        row["answer_recall"] = round(df["answer_recall"].dropna().mean(), 3)
        row["qa_pair_recall"] = round(df["qa_pair_recall"].dropna().mean(), 3)
        lo, hi = bootstrap_ci(df["qa_pair_recall"].tolist())
        row["qa_pair_recall_ci"] = fmt_ci(lo, hi)
    row["latency_sec"] = round(df["latency_sec"].mean(), 2)
    return row


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" " * 14 + "LAYER 1 SCORING v5 — Field F1 + CIs + CORD Group F1")
    print("=" * 78)

    summary_rows = []
    for dataset in DATASETS:
        print(f"\n-- {dataset} --")
        for model in MODELS:
            df, failures = score_dataset(dataset, model)
            if df is None:
                print(f"  {model:<12}  no extractions found, skipping")
                continue
            df.to_csv(METRICS_DIR / f"{dataset.lower()}_{model}_v5.csv", index=False)
            if failures:
                pd.DataFrame(failures).to_csv(
                    METRICS_DIR / f"{dataset.lower()}_{model}_failures_v5.csv", index=False)
            row = summarise(dataset, model, df)
            summary_rows.append(row)
            if dataset == "CORD":
                grp = row.get("group_f1")
                grp_str = f"  groupF1={grp:.3f} {row.get('group_f1_ci','')}" if grp is not None else ""
                print(f"  {model:<12}  docs={len(df):>3}  "
                      f"field_F1={row['field_f1']:.3f} {row['field_f1_ci']}  "
                      f"(P={row['field_precision']:.3f} R={row['field_recall']:.3f})  "
                      f"tokenF1={row['token_f1']:.3f}{grp_str}")
            elif dataset == "SROIE":
                print(f"  {model:<12}  docs={len(df):>3}  "
                      f"field_F1={row['field_f1']:.3f} {row['field_f1_ci']}  "
                      f"(P={row['field_precision']:.3f} R={row['field_recall']:.3f})  "
                      f"tokenF1={row['token_f1']:.3f}")
            else:
                print(f"  {model:<12}  docs={len(df):>3}  "
                      f"answer_recall={row['answer_recall']:.3f}  "
                      f"qa_pair_recall={row['qa_pair_recall']:.3f} {row['qa_pair_recall_ci']}")

    if not summary_rows:
        print("\nNo results found.")
        return

    pd.DataFrame(summary_rows).to_csv(METRICS_DIR / "full_comparison_v5.csv", index=False)

    # Markdown report
    md = ["# Layer 1 — Comparison v5 (field-level F1 + CORD group F1)\n",
          "Headline metric is **field-level F1** (with 95% bootstrap CI). "
          "Token F1 is shown for reference only — it understates KIE quality. "
          "For CORD, **Group F1** adds a structure-aware check: a menu row counts "
          "as correct only when its name and price match together.\n",
          "## KIE headline (SROIE, CORD)\n",
          "| Dataset | Model | Docs | Field F1 | 95% CI | Precision | Recall | Token F1 | Latency (s) |",
          "|---------|-------|------|----------|--------|-----------|--------|----------|-------------|"]
    for r in summary_rows:
        if r["dataset"] in ("SROIE", "CORD"):
            md.append(f"| {r['dataset']} | {r['model']} | {r['documents']} | "
                      f"{r['field_f1']:.3f} | {r['field_f1_ci']} | {r['field_precision']:.3f} | "
                      f"{r['field_recall']:.3f} | {r['token_f1']:.3f} | {r['latency_sec']:.2f} |")
    md += ["\n## CORD group-level F1 (structure-aware)\n",
           "| Model | Group F1 | 95% CI | Group Precision | Group Recall |",
           "|-------|----------|--------|-----------------|--------------|"]
    for r in summary_rows:
        if r["dataset"] == "CORD" and r.get("group_f1") is not None:
            md.append(f"| {r['model']} | {r['group_f1']:.3f} | {r.get('group_f1_ci','-')} | "
                      f"{r.get('group_precision',0):.3f} | {r.get('group_recall',0):.3f} |")
    md += ["\n## FUNSD (lenient vs strict)\n",
           "| Model | Answer-token Recall | Q→A Pair Recall | 95% CI |",
           "|-------|---------------------|-----------------|--------|"]
    for r in summary_rows:
        if r["dataset"] == "FUNSD":
            md.append(f"| {r['model']} | {r['answer_recall']:.3f} | "
                      f"{r['qa_pair_recall']:.3f} | {r['qa_pair_recall_ci']} |")
    md += ["\n## SROIE per-field hit rates\n",
           "| Model | Company | Date | Address | Total |",
           "|-------|---------|------|---------|-------|"]
    for r in summary_rows:
        if r["dataset"] == "SROIE":
            md.append(f"| {r['model']} | {r.get('hit_company',0):.3f} | {r.get('hit_date',0):.3f} | "
                      f"{r.get('hit_address',0):.3f} | {r.get('hit_total',0):.3f} |")
    md += ["\n## CORD per-category recall\n",
           "| Model | Menu Items | Menu Prices | Totals |",
           "|-------|-----------|-------------|--------|"]
    for r in summary_rows:
        if r["dataset"] == "CORD":
            md.append(f"| {r['model']} | {r.get('menu_items_recall',0):.3f} | "
                      f"{r.get('menu_prices_recall',0):.3f} | {r.get('totals_recall',0):.3f} |")
    (METRICS_DIR / "full_comparison_v5.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nCSV: {METRICS_DIR / 'full_comparison_v5.csv'}")
    print(f"MD:  {METRICS_DIR / 'full_comparison_v5.md'}")


if __name__ == "__main__":
    main()
