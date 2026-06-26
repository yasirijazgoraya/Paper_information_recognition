# Layer 1 v5 — deployment

Adds CORD group-level F1 on top of v4. Reads cached extractions; no model
re-runs.

## 1. Add the scorer

Place `score_v5.py` at:

    code/layer1_vision/score_v5.py

It does not modify v4; both can coexist.

## 2. Run

    cd /mnt/yasir_drive/E_DATA/ResearchProject
    python code/layer1_vision/score_v5.py

Writes:
- `results/metrics/full_comparison_v5.csv`
- `results/metrics/full_comparison_v5.md`
- `results/metrics/<dataset>_<model>_v5.csv`
- `results/metrics/<dataset>_<model>_failures_v5.csv`

The CORD section now includes a Group F1 table.

## 3. Update the docs

- Append `METRICS_v5_addendum.md` to `METRICS.md` (adds section 11).
- Append `REPORT_v5_entry.md` to `REPORT.md`.
- Add the CORD Group F1 column to `RESULTS.md` once the run completes.

## Summary of the change

Group F1 scores a CORD menu row as a single unit (name and price together),
adding a structure-aware view that field-level F1 does not provide. Reported for
CORD only; SROIE and FUNSD are unchanged.
