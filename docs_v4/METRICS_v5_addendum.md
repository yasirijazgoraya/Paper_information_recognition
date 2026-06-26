# Evaluation Metrics — v5 Addendum (CORD group-level F1)

This addendum supplements `METRICS.md` and the v4 addendum. It documents the
group-level metric added in `score_v5.py`. All v4 metrics are retained
unchanged. v5 adds one metric, for CORD only.

## 11. CORD Group F1 (structure-aware)

CORD menu lines are grouped: a menu row links its name, price, and quantity
through a shared `group_id` in the annotation. Field-level F1 scores each value
independently and therefore does not verify whether values are assembled into
the correct row. A model can recover every value yet associate a name with the
wrong price; field-level F1 does not detect this, but it is a meaningful error
in practice.

Group F1 treats a whole menu row as a single unit. A ground-truth row counts as
recovered only when its name and price are both present together in the
prediction. Group-level precision, recall, and F1 are then computed over rows,
in the same precision/recall manner as the field-level metric. This follows the
group-level evaluation described in KIEval (Khang et al., 2025).

### Scope

Group F1 is reported for CORD only. SROIE and FUNSD contain no grouped entities,
so field-level and relation-level F1 remain complete for them and are unchanged.

### Reading the two CORD numbers together

- High field F1 with lower group F1 indicates that values are recovered but not
  always grouped into the correct rows.
- When the two are close, the model both extracts and groups consistently.

### Implementation notes

- Ground-truth rows are reconstructed from `valid_line` entries sharing a
  `group_id`, keeping only menu groups (sub-total and total are not menu rows).
- The cached predictions are flat text, so a row is counted as recovered when
  its name and price co-occur in the prediction. Group precision is estimated
  against the number of priced lines in the prediction.
- Because the prediction side is approximate, group F1 is reported as a
  structure-aware indicator alongside field F1, not as a replacement for it.

### Verification

The metric was checked on cached extractions before integration. When a menu
row's price is altered so the name no longer pairs with the correct price, field
recall can remain high (the value still appears elsewhere on the receipt) while
group F1 falls — confirming that group F1 captures grouping errors that
field-level F1 does not.
