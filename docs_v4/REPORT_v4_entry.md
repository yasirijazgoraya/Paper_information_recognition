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
