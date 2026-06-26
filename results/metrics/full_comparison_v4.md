# Layer 1 — Field-Level Comparison v4

Headline metric is **field-level F1** (with 95% bootstrap CI). Token F1 is shown for reference only — it understates KIE quality.

## KIE headline (SROIE, CORD)

| Dataset | Model | Docs | Field F1 | 95% CI | Precision | Recall | Token F1 | Latency (s) |
|---------|-------|------|----------|--------|-----------|--------|----------|-------------|
| SROIE | tesseract | 347 | 0.596 | [0.571, 0.618] | 0.518 | 0.719 | 0.268 | 0.56 |
| SROIE | paddleocr | 347 | 0.576 | [0.559, 0.593] | 0.510 | 0.666 | 0.229 | 1.21 |
| SROIE | qwen2vl | 347 | 0.817 | [0.807, 0.827] | 0.717 | 0.958 | 0.349 | 8.46 |
| CORD | tesseract | 100 | 0.228 | [0.188, 0.265] | 0.213 | 0.281 | 0.379 | 0.63 |
| CORD | paddleocr | 100 | 0.445 | [0.428, 0.463] | 0.340 | 0.657 | 0.820 | 0.48 |
| CORD | qwen2vl | 100 | 0.438 | [0.399, 0.472] | 0.380 | 0.564 | 0.766 | 3.69 |

## FUNSD (lenient vs strict)

| Model | Answer-token Recall | Q→A Pair Recall | 95% CI |
|-------|---------------------|-----------------|--------|
| tesseract | 0.643 | 0.168 | [0.121, 0.223] |
| paddleocr | 0.610 | 0.222 | [0.166, 0.278] |
| qwen2vl | 0.799 | 0.336 | [0.267, 0.402] |

## SROIE per-field hit rates

| Model | Company | Date | Address | Total |
|-------|---------|------|---------|-------|
| tesseract | 0.585 | 0.718 | 0.793 | 0.781 |
| paddleocr | 0.510 | 0.914 | 0.265 | 0.977 |
| qwen2vl | 0.928 | 0.960 | 0.968 | 0.977 |

## CORD per-category recall

| Model | Menu Items | Menu Prices | Totals |
|-------|-----------|-------------|--------|
| tesseract | 0.452 | 0.325 | 0.155 |
| paddleocr | 0.862 | 0.952 | 0.450 |
| qwen2vl | 0.778 | 0.766 | 0.382 |
