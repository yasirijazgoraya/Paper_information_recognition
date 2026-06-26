# Progress Report

A running log of project setup, data preparation, and next steps.

## 1. Infrastructure

- **Server:** `deepblue` (192.168.1.35), Ubuntu 24.04.1 LTS, kernel 6.8.0-111
- **Access:** SSH from Windows workstation (`ssh sysadmin@192.168.1.35`)
- **Storage:** project lives on `/mnt/yasir_drive/` (large attached drive)
- **Python:** Miniconda3, env `edata`

## 2. Directory setup

Created the project data root on the server:

```bash
mkdir -p /mnt/yasir_drive/E_DATA
```

Project skeleton (already present under `ResearchProject/`):

```
ResearchProject/
├── code/
├── data/
├── logs/
├── notebooks/
├── results/
├── requirements.txt
└── README.md
```

## 3. Data transfer

Source: `D:\E-DATA\Research\Datasets\compressed` on the Windows workstation.
Destination: `/mnt/yasir_drive/E_DATA/compressed/` on the server.

Transferred with `scp` from PowerShell:

```powershell
scp -r D:\E-DATA\Research\Datasets\compressed sysadmin@192.168.1.35:/mnt/yasir_drive/E_DATA/
```

Archives received:

```
compressed/
├── CORD.zip
├── FUNSD.zip
├── RVL-CDIP.zip
└── SROIE.zip
```

## 4. Extraction

Extracted all four archives into the project's `data/` directory:

```bash
cd /mnt/yasir_drive/E_DATA
for f in compressed/*.zip; do
    unzip -q "$f" -d ResearchProject/data/
done
```

Verification:

```bash
ls -lh ResearchProject/data/
du -sh ResearchProject/data/*
```

Original zips retained under `compressed/` as a backup.

## 5. Next steps — OCR & document understanding pipeline

### 5.1 Preprocessing

- Inspect each dataset's structure (image dirs, annotation format) and document it under `notebooks/`.
- Build a unified annotation schema across CORD / FUNSD / SROIE (JSON with `words`, `bboxes`, `labels`).
- Generate consistent train/val/test splits; for datasets that ship with official splits (CORD, FUNSD, SROIE) preserve them.
- Sanity-check image counts and class balances (especially RVL-CDIP's 16 classes).

### 5.2 Local OCR backend

- Evaluate candidates: **PaddleOCR**, **Tesseract 5**, **TrOCR**.
- Run a small benchmark (speed, CER/WER) on a held-out sample of SROIE and CORD receipts.
- Pick the backend, wrap it in `code/ocr/` with a simple API: `image -> [{text, bbox, conf}]`.
- Cache OCR outputs to disk to avoid re-running during model training.

### 5.3 Models

- **KIE / form understanding (CORD, FUNSD, SROIE):** fine-tune **LayoutLMv3** as the primary baseline; consider **Donut** for end-to-end OCR-free comparison.
- **Document classification (RVL-CDIP):** **DiT** or **ViT** baseline, 16-class softmax.
- Training scripts under `code/train/`, configs under `code/configs/`.

### 5.4 Evaluation

| Dataset  | Metric                                    |
|----------|-------------------------------------------|
| CORD     | Entity-level F1                           |
| FUNSD    | Entity-level F1, relation extraction F1   |
| SROIE    | Field-level F1 (per task)                 |
| RVL-CDIP | Top-1 accuracy                            |

All runs logged under `logs/`; final metrics summarized in `results/`.

### 5.5 Reporting

- Per-dataset error analysis notebooks under `notebooks/`.
- Final tables and figures in `results/`.
- This `REPORT.md` updated after each milestone.

## 6. Open items

- Confirm research direction (general benchmarking vs. a specific method).
- Decide whether to include a domain-adaptation track (the server already holds a `Domain_adaptation/` directory at `/mnt/yasir_drive/`).
- Set up experiment tracking (Weights & Biases or local TensorBoard).
# Milestone: Layer 1 scoring v4 (field-level F1)

## What changed
Upgraded Layer 1 scoring from v3 to v4. Headline KIE metric for SROIE and CORD
is now field-level F1 with a 95% bootstrap CI, instead of recall-only. v3
field_recall could not see false positives, so an over-emitting backend scored
artificially well. v4 adds precision and reports the harmonic mean - the same
quantity as the SROIE Task 3 leaderboard Hmean, so results are now comparable.

Token F1 retained but demoted to a supporting column. It was previously first
and made results look weak (~0.35) despite strong per-field hit rates
(~0.93-0.98). A reporting artefact, not a model weakness.

All v3 matching preserved: Fix A (Indonesian thousands), Fix B (fuzzy address),
Fix C (FUNSD Q->A). FUNSD scoring unchanged.

## Verified
CORD receipt_00000 (PaddleOCR): recall 1.000 but precision 0.278 => F1 0.435.
v3 would report 1.000 and hide the over-emission. SROIE X00016469670: company,
date, total matched; address missed (fuzzy overlap < 0.70) => F1 0.750.

## Next steps
1. Run score_v4.py over all extractions; regenerate full_comparison_v4.{csv,md}.
2. Update RESULTS.md to lead with the v4 field-F1 table.
3. RQ1 Step 2: add Textract, Claude Vision, GPT-4o; re-score same harness.
FUNSD zero-shot added: kv_f1 0.496 (P 0.488 / R 0.517)
