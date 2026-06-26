# RESULTS.md - v4 update template

Replace the headline table in RESULTS.md with the structure below. Numbers fill
automatically from results/metrics/full_comparison_v4.md. Field-level F1 leads;
Token F1 is a supporting column; every KIE number carries a 95% CI.

## Layer 1 - OCR Backend Evaluation (v4)

Headline KIE metric is field-level F1 (precision + recall, harmonic mean),
reported with 95% bootstrap CI, comparable to the SROIE Task 3 leaderboard.
Token F1 shown for reference only - it understates KIE quality.

### KIE headline (SROIE, CORD)

| Dataset | Backend | Docs | Field F1 | 95% CI | Precision | Recall | Token F1 | Latency (s/page) |
|---------|---------|------|----------|--------|-----------|--------|----------|------------------|
| SROIE | Tesseract | 347 | ... | ... | ... | ... | ... | ... |
| SROIE | PaddleOCR | 347 | ... | ... | ... | ... | ... | ... |
| SROIE | Qwen2-VL  | 347 | ... | ... | ... | ... | ... | ... |
| CORD  | Tesseract | 100 | ... | ... | ... | ... | ... | ... |
| CORD  | PaddleOCR | 100 | ... | ... | ... | ... | ... | ... |
| CORD  | Qwen2-VL  | 100 | ... | ... | ... | ... | ... | ... |

### FUNSD - lenient vs strict (unchanged from v3)

| Backend | Answer-token Recall | Q->A Pair Recall | 95% CI |
|---------|---------------------|------------------|--------|
| Tesseract | ... | ... | ... |
| PaddleOCR | ... | ... | ... |
| Qwen2-VL  | ... | ... | ... |

### How to read
- Field F1 is the comparison metric. High recall + low precision now gives a
  moderate F1 - over-emission is penalised.
- Overlapping CIs => treat the difference as noise.
- Low Token F1 + high Field F1 => right fields recovered, extra text emitted.
- Label fine-tuned vs zero-shot. StrucTexT (98.7%) is a cited fine-tuned ceiling.
