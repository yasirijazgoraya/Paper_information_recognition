# RESULTS.md — Layer 1 OCR Backend Evaluation (v4)

Comparison of three OCR/VLM backends — Tesseract 5, PaddleOCR, Qwen2-VL — on
three document-understanding benchmarks: FUNSD (forms), SROIE (receipts), CORD
(receipts, Indonesian).

The headline KIE metric is **field-level F1** (precision + recall, harmonic
mean), reported with a 95% bootstrap confidence interval and directly comparable
to the SROIE Task 3 leaderboard. Token F1 is shown for reference only — it
understates KIE quality because it penalises any extra emitted words. All v3
matching logic (Fix A Indonesian numbers, Fix B fuzzy address, Fix C FUNSD Q→A)
is unchanged; v4 adds precision, symmetric F1, and confidence intervals.

## KIE headline (SROIE, CORD)

| Dataset | Backend | Docs | Field F1 | 95% CI | Precision | Recall | Token F1 | Latency (s/page) |
|---------|---------|------|----------|----------------|-----------|--------|----------|------------------|
| SROIE | Tesseract | 347 | 0.596 | [0.571, 0.618] | 0.518 | 0.719 | 0.268 | 0.56 |
| SROIE | PaddleOCR | 347 | 0.576 | [0.559, 0.593] | 0.510 | 0.666 | 0.229 | 1.21 |
| SROIE | **Qwen2-VL** | 347 | **0.817** | [0.807, 0.827] | 0.717 | 0.958 | 0.349 | 8.46 |
| CORD | Tesseract | 100 | 0.228 | [0.188, 0.265] | 0.213 | 0.281 | 0.379 | 0.63 |
| CORD | **PaddleOCR** | 100 | **0.445** | [0.428, 0.463] | 0.340 | 0.657 | 0.820 | 0.48 |
| CORD | Qwen2-VL | 100 | 0.438 | [0.399, 0.472] | 0.380 | 0.564 | 0.766 | 3.69 |

## FUNSD — lenient vs strict

| Backend | Answer-token Recall | Q→A Pair Recall | 95% CI |
|---------|---------------------|-----------------|----------------|
| Tesseract | 0.643 | 0.168 | [0.121, 0.223] |
| PaddleOCR | 0.610 | 0.222 | [0.166, 0.278] |
| **Qwen2-VL** | **0.799** | **0.336** | [0.267, 0.402] |

## SROIE — per-field hit rates

| Backend | Company | Date | Address | Total |
|---------|---------|------|---------|-------|
| Tesseract | 0.585 | 0.718 | 0.793 | 0.781 |
| PaddleOCR | 0.510 | 0.914 | 0.265 | 0.977 |
| Qwen2-VL | 0.928 | 0.960 | 0.968 | 0.977 |

## CORD — per-category recall

| Backend | Menu Items | Menu Prices | Totals |
|---------|-----------|-------------|--------|
| Tesseract | 0.452 | 0.325 | 0.155 |
| PaddleOCR | 0.862 | 0.952 | 0.450 |
| Qwen2-VL | 0.778 | 0.766 | 0.382 |

## Findings

**Qwen2-VL is the strongest on SROIE by a clear margin.** Its field F1 of 0.817
sits well above Tesseract (0.596) and PaddleOCR (0.576), and the confidence
intervals do not overlap — the gap is statistically real, not noise. On the
old Token-F1 headline this same model scored only 0.349; field-level F1 reveals
the actual KIE quality that token-level scoring obscured.

**CORD is effectively a tie between PaddleOCR (0.445) and Qwen2-VL (0.438).**
Their confidence intervals overlap ([0.428, 0.463] vs [0.399, 0.472]), so the
difference is within noise. PaddleOCR is the better operational choice on CORD
because it is roughly 8× faster (0.48 vs 3.69 s/page).

**Precision and recall diverge in a way recall-only scoring hid.** Qwen2-VL on
SROIE recovers almost everything (R=0.958) but emits extra material (P=0.717);
on CORD all backends show precision well below recall, because receipts repeat
many numeric strings that count as false-positive candidates. This is now
visible in the headline and is the main reason CORD field-F1 looks lower than
the per-category recall.

**Strict structural scoring still exposes a gap.** FUNSD Q→A pair recall tops
out at 0.336 (Qwen2-VL) — even the best backend links only a third of
question-answer pairs correctly. This continues to motivate a layout-aware
extraction step (LayoutLMv3 / Donut) downstream.

## Notes on fairness and interpretation

- **CORD precision is conservative by construction.** Every stray numeric token
  counts as a false positive, so CORD field-F1 understates menu-extraction
  quality; the per-category recall table is the more flattering and arguably
  fairer view for that use case. See METRICS.md §10.
- **Fine-tuned vs zero-shot.** These three backends are zero-shot. When the
  cloud models (Textract, Claude Vision, GPT-4o) and a fine-tuned baseline
  (LayoutLMv3) are added, the fine-tuned/zero-shot distinction must be labelled.
  The SROIE Task 3 leaderboard SOTA (StrucTexT, 98.7% F1) is a fine-tuned
  reference ceiling, cited only — not placed in this table.

## Reproducibility

- Scorer: `code/layer1_vision/score_v4.py`
- Aggregate: `results/metrics/full_comparison_v4.{csv,md}`
- Per-doc: `results/metrics/<dataset>_<model>_v4.csv`
- Failures: `results/metrics/<dataset>_<model>_failures_v4.csv`
- Metric definitions: `METRICS.md` + `docs_v4/METRICS_v4_addendum.md`
