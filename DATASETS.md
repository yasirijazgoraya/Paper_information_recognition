# Datasets

The benchmarks used in this project, where they live on disk, and what the evaluation splits look like. All archives are kept at `/mnt/yasir_drive/E_DATA/compressed/`; extracted contents live under `ResearchProject/data/`.

## Summary

| Dataset  | Task                                | Train  | Val   | Test   | Used here (Layer 1) |
|----------|-------------------------------------|-------:|------:|-------:|---------------------|
| FUNSD    | Form understanding (entity + rel.)  |    149 |     — |     50 | 50 (full test)      |
| CORD     | Receipt KIE                         |    800 |   100 |    100 | 100 (full test)     |
| SROIE    | Scanned receipt OCR & KIE           |    626 |     — |    347 | 347 (full test)     |
| RVL-CDIP | Document image classification (16)  | 320 k  | 40 k  |  40 k  | not yet evaluated   |
| DocVQA   | Document visual question answering  | 10 194 | 1 286 |  1 287 | extractions only    |

Numbers refer to the standard public splits shipped with each release.

---

## FUNSD — Form Understanding in Noisy Scanned Documents

- **Task.** Word-level OCR plus entity labelling (`header`, `question`, `answer`, `other`) and relation extraction (question → answer links).
- **Size.** 199 fully-annotated scanned forms; 149 train, 50 test.
- **Annotations.** One JSON per image with word-level bounding boxes, text, entity label, and linking IDs.
- **Why included.** Stresses layout understanding on noisy, real-world forms (faxes, tax documents). Small enough to iterate quickly.
- **On disk.** `data/FUNSD/dataset/training_data/` and `data/FUNSD/dataset/testing_data/`, each with `images/` and `annotations/`.
- **Reference.** Jaume et al., *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents*, ICDAR-OST 2019.

## CORD — Consolidated Receipt Dataset

- **Task.** Key-information extraction from receipts: 30 fine-grained fields organised hierarchically (menu items with name / quantity / price, subtotal, total, etc.).
- **Size.** 1000 receipts in the public release; 800 train / 100 validation / 100 test.
- **Language / format.** Indonesian text, mostly thermal-printer fonts. Annotations are JSON with quad bounding boxes and nested field categories.
- **Quirk relevant to scoring.** Prices use periods as thousands separators (`25.000` = 25 000). Handled by Fix A in `score_v3.py` — see [METRICS.md](METRICS.md) §3.
- **On disk.** `data/CORD/{train,dev,test}/{image,json}/`.
- **Reference.** Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, NeurIPS Document Intelligence Workshop, 2019.

## SROIE — Scanned Receipts OCR and Information Extraction

- **Task.** ICDAR 2019 competition with three tracks: (1) text localization, (2) OCR transcription, (3) KIE over four fields — `company`, `date`, `address`, `total`. We use track 3.
- **Size.** 626 train + 347 test receipts (English, store-printed).
- **Annotations.** Per-image text file with quadrilateral coordinates and transcription, plus a JSON with the four KIE fields.
- **Quirk relevant to scoring.** Addresses are multi-line; OCR engines that emit lines independently fail strict exact-match. Handled by Fix B (70 % token-set overlap on address) — see [METRICS.md](METRICS.md) §2.
- **On disk.** `data/SROIE/{train,test}/img/`, `data/SROIE/{train,test}/box/`, `data/SROIE/{train,test}/entities/`.
- **Reference.** Huang et al., *ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction*, ICDAR 2019.

## RVL-CDIP — Ryerson Vision Lab Complex Document Information Processing

- **Task.** 16-class document image classification (letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, memo).
- **Size.** 400 000 grayscale images; 320 k train / 40 k val / 40 k test. Drawn from the IIT-CDIP / Tobacco legal-discovery corpus.
- **Why included.** Single-label classification at scale — a useful counterpoint to the structured-extraction benchmarks above.
- **On disk.** `data/RVL-CDIP/images/` plus three label files (`train.txt`, `val.txt`, `test.txt`), one image-path-and-class per line.
- **Status.** Not part of the Layer 1 OCR comparison. Reserved for a downstream classification baseline (ViT / DiT).
- **Reference.** Harley, Ufkes & Derpanis, *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval*, ICDAR 2015.

## DocVQA — Document Visual Question Answering

- **Task.** Answer natural-language questions grounded in a single document image. Free-form text answers.
- **Size (Single-Page DocVQA).** 10 194 train / 1 286 val / 1 287 test images; ~50 000 QA pairs total.
- **Why included.** Tests the OCR backends in a question-answering setting rather than fixed-schema KIE.
- **On disk.** `data/DocVQA/` with images and a per-split JSON of QA pairs.
- **Status.** OCR extractions produced (`results/extractions/DocVQA/`); scoring not yet wired into `score_v3.py`.
- **Reference.** Mathew, Karatzas & Jawahar, *DocVQA: A Dataset for VQA on Document Images*, WACV 2021.

---

## Licensing & redistribution

All datasets are used **only for academic research**. None are committed to the repository (see `.gitignore`). To reproduce, download each from its official source:

- FUNSD: https://guillaumejaume.github.io/FUNSD/
- CORD: https://github.com/clovaai/cord
- SROIE: https://rrc.cvc.uab.es/?ch=13
- RVL-CDIP: https://huggingface.co/datasets/aharley/rvl_cdip (mirrors the original release)
- DocVQA: https://www.docvqa.org/

Refer to each project's license before redistributing.
