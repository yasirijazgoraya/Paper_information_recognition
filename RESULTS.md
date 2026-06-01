# Layer 1 — OCR Backend Evaluation

Comparison of three OCR backends — **Tesseract 5**, **PaddleOCR**, **Qwen2-VL** — on three document-understanding benchmarks: FUNSD (forms), SROIE (receipts, ICDAR 2019), and CORD (receipts, Indonesian).

Metrics use the v3 scoring methodology:
- CORD numeric matching handles the Indonesian thousands separator (`25.000` → 25 000).
- SROIE address matching uses 70 % token-set overlap to tolerate multi-line layouts.
- FUNSD adds a strict question→answer pair recall on top of lenient token recall.

## Headline numbers

| Dataset | Backend       | Docs | Token F1  | Field / Key Recall | Latency (s/page) |
|---------|---------------|-----:|----------:|-------------------:|-----------------:|
| FUNSD   | Tesseract     |   50 |     0.630 |              0.643 |             0.33 |
| FUNSD   | PaddleOCR     |   50 |     0.686 |              0.610 |             1.58 |
| FUNSD   | **Qwen2-VL**  |   50 | **0.854** |          **0.799** |             7.32 |
| SROIE   | Tesseract     |  347 |     0.268 |              0.719 |             0.56 |
| SROIE   | PaddleOCR     |  347 |     0.229 |              0.666 |             1.21 |
| SROIE   | **Qwen2-VL**  |  347 | **0.349** |          **0.958** |             8.46 |
| CORD    | Tesseract     |  100 |     0.379 |              0.281 |             0.63 |
| CORD    | **PaddleOCR** |  100 | **0.820** |          **0.657** |             0.48 |
| CORD    | Qwen2-VL      |  100 |     0.766 |              0.564 |             3.69 |

## Per-dataset detail

### SROIE — field hit rates

| Backend   |   Company |      Date |   Address |     Total |
|-----------|----------:|----------:|----------:|----------:|
| Tesseract |     0.585 |     0.718 |     0.793 |     0.781 |
| PaddleOCR |     0.510 |     0.914 |     0.265 |     0.977 |
| Qwen2-VL  | **0.928** | **0.960** | **0.968** | **0.977** |

Qwen2-VL is the only backend above 0.9 on every field. PaddleOCR's address recall collapses (0.265) because it splits multi-line addresses across separate detections.

### CORD — field recall

| Backend   | Menu items | Menu prices |    Totals |
|-----------|-----------:|------------:|----------:|
| Tesseract |      0.452 |       0.325 |     0.155 |
| PaddleOCR |  **0.862** |   **0.952** | **0.450** |
| Qwen2-VL  |      0.778 |       0.766 |     0.382 |

PaddleOCR is the clear winner on CORD — both the fastest backend (0.48 s/page) and the most accurate on prices and totals. Tesseract fails on small-font price columns; Qwen2-VL is competitive but several points behind on every field.

### FUNSD — lenient vs. strict

| Backend   | Answer-token recall | Q → A pair recall |
|-----------|--------------------:|------------------:|
| Tesseract |               0.643 |             0.168 |
| PaddleOCR |               0.610 |             0.222 |
| Qwen2-VL  |           **0.799** |         **0.336** |

All three drop sharply when scoring requires the question and its answer to be linked, not just both present. Qwen2-VL is the strongest, but even it pairs only one third of questions correctly — confirming that an extra layout / relation step is needed downstream.

## Takeaways

- **Qwen2-VL is the most accurate overall** — best on FUNSD and SROIE by a wide margin, second on CORD. The cost is latency: 7–8 s/page on the local GPU, roughly **5–15× slower** than Tesseract or PaddleOCR.
- **PaddleOCR is the best speed/quality trade-off for receipts.** On CORD it beats Qwen2-VL outright; on SROIE it ties on totals and dates. Its weak spot is multi-line addresses.
- **Tesseract is competitive only on Latin-script forms** (FUNSD). It collapses on small fonts and dense numeric tables (CORD totals: 15 %).
- **Strict structural scoring matters.** Token-level F1 looks healthy across the board, but FUNSD Q→A pair recall and CORD totals recall expose how much information is still missed — motivating layout-aware extraction (LayoutLMv3 / Donut) as the next step.

## Operational recommendation

| Use case                          | Backend     | Reason                                  |
|-----------------------------------|-------------|-----------------------------------------|
| High-throughput receipts          | PaddleOCR   | Best CORD quality, fastest of the three |
| High-accuracy receipts / forms    | Qwen2-VL    | Best SROIE & FUNSD, multi-line robust   |
| Low-resource / CPU-only fallback  | Tesseract   | Cheapest, acceptable on clean forms     |

Default downstream pipeline: **PaddleOCR** for bulk OCR caching, with **Qwen2-VL** as a fallback on documents where Paddle's confidence is low or addresses are detected as multi-line.

## Reproducibility

- Raw per-document outputs: `results/extractions/{DATASET}/{backend}/*.json`
- Per-run metrics: `results/metrics/{dataset}_{backend}_v3.csv`
- Failure lists: `results/metrics/{dataset}_{backend}_failures.csv`
- Aggregate tables: `results/metrics/full_comparison_v3.{csv,md}`
- Scripts: `code/layer1_vision/day{1,2,3}_*.py`, `score_v3.py`
