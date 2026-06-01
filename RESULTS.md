# Results & Evaluation

Tracking sheet for all experiments. Tables are filled in as runs complete. Each row should link to a config / log / checkpoint under `logs/` or `results/`.

Legend:
- **TBD** — not yet run
- **—** — not applicable for this dataset/model
- All metrics are reported on the official test split unless noted.

---

## 1. OCR backend comparison (FUNSD)

Comparison of OCR engines on the FUNSD test set. Metrics: character error rate (CER), word error rate (WER), mean detection IoU vs. ground-truth boxes, and average inference time per page on the local GPU.

| Backend       | Version | CER ↓ | WER ↓ | Det. IoU ↑ | Time/page (s) ↓ | Script                                       |
|---------------|---------|-------|-------|------------|-----------------|----------------------------------------------|
| Tesseract     | 5.x     | TBD   | TBD   | TBD        | TBD             | `code/layer1_vision/day1_tesseract_funsd.py` |
| PaddleOCR     | TBD     | TBD   | TBD   | TBD        | TBD             | `code/layer1_vision/day2_paddleocr_funsd.py` |
| Qwen-VL (OCR) | TBD     | TBD   | TBD   | TBD        | TBD             | `code/layer1_vision/day3_qwen_funsd.py`      |

**Notes:**
- TBD — observations after backend selection (errors on small fonts, handwriting, rotation, etc.).
- Chosen backend for downstream tasks: **TBD**.

---

## 2. Per-dataset results

### 2.1 CORD — Receipt key-information extraction

Entity-level F1 on the CORD test split (1000 receipts, 30 entity types).

| Model       | OCR        | Precision | Recall | F1 ↑ | Notes |
|-------------|------------|-----------|--------|------|-------|
| LayoutLMv3  | TBD        | TBD       | TBD    | TBD  |       |
| Donut       | end-to-end | TBD       | TBD    | TBD  |       |
| (baseline)  | TBD        | TBD       | TBD    | TBD  |       |

### 2.2 FUNSD — Form understanding

Entity-level F1 and relation extraction F1 on the FUNSD test split (50 forms).

| Model       | OCR        | Entity F1 ↑ | Relation F1 ↑ | Notes |
|-------------|------------|-------------|----------------|-------|
| LayoutLMv3  | TBD        | TBD         | TBD            |       |
| Donut       | end-to-end | TBD         | —              |       |
| (baseline)  | TBD        | TBD         | TBD            |       |

### 2.3 SROIE — Scanned receipt OCR & KIE

Field-level F1 on the four target fields (company, date, address, total).

| Model       | OCR        | Company | Date | Address | Total | Macro F1 ↑ |
|-------------|------------|---------|------|---------|-------|------------|
| LayoutLMv3  | TBD        | TBD     | TBD  | TBD     | TBD   | TBD        |
| Donut       | end-to-end | TBD     | TBD  | TBD     | TBD   | TBD        |
| (baseline)  | TBD        | TBD     | TBD  | TBD     | TBD   | TBD        |

### 2.4 RVL-CDIP — Document image classification

Top-1 accuracy on the official test split (40k images, 16 classes).

| Model       | Pretrain        | Top-1 Acc ↑ | Macro F1 ↑ | Notes |
|-------------|-----------------|-------------|------------|-------|
| ViT-Base    | ImageNet-21k    | TBD         | TBD        |       |
| DiT-Base    | IIT-CDIP        | TBD         | TBD        |       |
| (baseline)  | TBD             | TBD         | TBD        |       |

---

## 3. Cross-dataset summary

Headline numbers for the report. Update once §1–§2 are filled in.

| Dataset  | Task                 | Best model | Metric     | Score |
|----------|----------------------|------------|------------|-------|
| CORD     | KIE                  | TBD        | Entity F1  | TBD   |
| FUNSD    | Form understanding   | TBD        | Entity F1  | TBD   |
| SROIE    | Receipt KIE          | TBD        | Macro F1   | TBD   |
| RVL-CDIP | Doc classification   | TBD        | Top-1 Acc  | TBD   |

---

## 4. Ablations & error analysis

To populate after baselines are done. Suggested studies:

- **OCR sensitivity** — re-run KIE models with each OCR backend; report Δ F1.
- **Layout vs. text-only** — strip layout features from LayoutLMv3; quantify the gap.
- **Data scaling** — train on 25 / 50 / 100 % of training data; plot learning curves.
- **Error buckets** — confusion matrices for RVL-CDIP; common failure modes for FUNSD relations.

---

## 5. Reproducibility

For each filled-in row above, record:

- Commit hash of the code used.
- Config file path (`code/configs/...`).
- Log directory (`logs/<run_name>/`).
- Checkpoint location (`results/checkpoints/<run_name>/`).
- Hardware (GPU model, CUDA version).

This makes every number in the tables traceable to an exact run.
