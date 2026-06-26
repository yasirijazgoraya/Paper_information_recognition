# Layer 1 v4 — deployment

Field-level F1 (precision + recall + 95% CI) replaces recall-only as the
headline KIE metric for SROIE and CORD. Token F1 demoted to a supporting
column. All v3 matching (Fix A/B/C) unchanged. Run:

    python code/layer1_vision/score_v4.py

Outputs full_comparison_v4.{csv,md} plus per-doc CSVs. No model re-runs.
