# Milestone: Layer 1 scoring v5 (CORD group-level F1)

## What changed

Added `score_v5.py`, which extends v4 with a structure-aware metric for CORD.
All v4 metrics are retained unchanged.

CORD menu lines are grouped (name, price, and quantity share a `group_id`).
Field-level F1 scores each value independently and does not check whether values
are assembled into the correct row. v5 adds Group F1, which treats a whole menu
row as one unit: a row counts as correct only when its name and price match
together. This follows the group-level evaluation described in KIEval (2025).

Group F1 is reported for CORD only. SROIE and FUNSD have no grouped entities, so
their scoring is unchanged.

## Why it matters

Field F1 alone can report high scores while missing grouping errors that affect
downstream use (for example, storing a name with the wrong price in a database).
Reporting Group F1 alongside Field F1 makes the CORD evaluation complete and
explicit about structure.

## How it was verified

The metric was checked on cached extractions before integration. With a correct
menu row, Group F1 is 1.0. When the row's price is altered so the name no longer
pairs with the correct price, field recall can stay high (the value still
appears elsewhere on the receipt) while Group F1 falls — confirming the metric
captures grouping errors that field-level F1 does not.

## Next steps

1. Run `score_v5.py` over all cached extractions; regenerate
   `full_comparison_v5.{csv,md}`.
2. Add the CORD Group F1 column to `RESULTS.md`.
3. Continue with the cloud-model evaluation for the full RQ1 comparison.
