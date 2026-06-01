# ResearchProject — Document OCR & Understanding

A research project on document understanding and OCR using benchmark datasets (CORD, FUNSD, SROIE, RVL-CDIP). Work runs on a local Ubuntu 24.04 server with GPU acceleration; this repo tracks the code, configs, and results.

## Environment

- **Server:** Ubuntu 24.04.1 LTS, kernel 6.8.0
- **Conda env:** `edata`
- **Data root:** `/mnt/yasir_drive/E_DATA/`
- **Project root:** `/mnt/yasir_drive/E_DATA/ResearchProject/`

## Repository layout

```
ResearchProject/
├── code/           # training, evaluation, preprocessing scripts
├── data/           # datasets (extracted; not tracked in git)
├── logs/           # training/eval logs
├── notebooks/      # exploration & analysis
├── results/        # checkpoints, metrics, figures
├── requirements.txt
├── README.md
└── REPORT.md       # progress log
```

## Datasets

All datasets live under `ResearchProject/data/` after extraction. Compressed archives are kept at `/mnt/yasir_drive/E_DATA/compressed/` as a backup.

| Dataset  | Task                                         | Notes                              |
|----------|----------------------------------------------|------------------------------------|
| CORD     | Receipt key-information extraction           | Indonesian receipts, ~1k images    |
| FUNSD    | Form understanding (entity + relation)       | 199 noisy scanned forms            |
| SROIE    | Scanned receipt OCR & information extraction | ICDAR 2019 competition data        |
| RVL-CDIP | Document image classification (16 classes)   | 400k images, ~37 GB extracted      |

Datasets are not committed to git. See REPORT.md for transfer and extraction steps.

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd ResearchProject

# 2. Conda env
conda create -n edata python=3.10 -y
conda activate edata
pip install -r requirements.txt

# 3. Point to data
export DATA_ROOT=/mnt/yasir_drive/E_DATA/ResearchProject/data
```

## Pipeline (planned)

1. **Preprocessing** — normalize image sizes, generate train/val/test splits per dataset, build a unified annotation schema.
2. **OCR** — local OCR backend (PaddleOCR / Tesseract / TrOCR) producing word-level boxes and text.
3. **Layout & KIE** — LayoutLMv3 (or Donut for end-to-end) fine-tuned per dataset.
4. **Classification (RVL-CDIP)** — ViT / DiT baseline for the 16-class task.
5. **Evaluation** — per-dataset metrics (F1 for FUNSD/SROIE/CORD entities, accuracy for RVL-CDIP).
6. **Reporting** — ablations and error analysis in `notebooks/`, final tables in `results/`.

## Status

See [REPORT.md](REPORT.md) for the current progress log.

## License

TBD.
