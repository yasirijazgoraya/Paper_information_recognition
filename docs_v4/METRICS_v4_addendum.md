# Evaluation Metrics — v4 Addendum

Supplements METRICS.md. Documents score_v4.py and resolves three of the four
caveats in the original section 7. All v3 matching logic (Fix A numeric
normalisation, Fix B fuzzy address overlap, Fix C FUNSD Q->A pairing) is
unchanged; v4 only changes how field-level scores are aggregated.

## 8. Field-level F1 (new headline, SROIE and CORD)

v3 reported field_recall only. A backend that dumps the whole page scores high
on recall while flooding output with noise, because recall never sees false
positives. v4 adds precision and reports the harmonic mean.

  Precision_d = tp / (tp + fp)
  Recall_d    = tp / (tp + fn)
  F1_d        = 2*P*R / (P + R)

where tp = GT field values matched, fn = GT values not matched, fp = predicted
values with no GT correspondent. Reported as macro-average over the split.
Field-level F1 is now the headline KIE number and is directly comparable to the
SROIE Task 3 leaderboard (its Hmean column is the same quantity). Token F1 is
demoted to a supporting column.

### SROIE one value per field (resolves section 7 third bullet)
Exactly one predicted value is selected per field before matching:
date = first date-like token; total = largest numeric value; company = first
non-empty line; address = the 1-3 lines after company. Same rule for every
backend. No plausible value => field counts as a miss.

### CORD symmetric F1 (resolves section 7 second bullet)
Precision computed against predicted candidates (text lines for menu items;
numeric tokens for prices/totals); symmetric F1 reported. Per-category recall
still reported for diagnosis.

## 9. Bootstrap confidence intervals (resolves section 7 first bullet)
Every headline metric reported with a 95% percentile bootstrap CI over
documents (2000 resamples, seed 42). Lets us say whether two backends differ or
are within noise, important for small splits (N=50 FUNSD, 100 CORD, 347 SROIE).

## 10. What v4 still does not capture
- OCR detection IoU: bbox quality still not scored, only text content.
- Calibrated confidence for generative VLMs: the confidences field is not used.
- SROIE selection-rule sensitivity: the one-value heuristic is simple, applied
  identically to all backends, documented here for transparency.
