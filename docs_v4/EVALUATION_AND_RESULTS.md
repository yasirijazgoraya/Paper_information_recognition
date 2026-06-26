# Evaluation, Results, and Method Notes — Stage 1 (Zero-shot vs Fine-tuned)

This document explains how the open-model arms of Stage 1 are evaluated, reports
the zero-shot and fine-tuned results on the receipt datasets (SROIE and CORD),
and records why LoRA was chosen for fine-tuning. It covers the two open-model
arms only: Qwen2.5-VL zero-shot and Qwen2.5-VL fine-tuned. The paid arms
(AWS Textract, Claude, GPT-4o) are scored separately on the same test split.

---

## 1. Evaluation metrics

Each model returns a structured record per document (a fixed receipt schema:
vendor, date, total, address, and a list of line items). Predictions are
compared field by field against the gold labels. Two kinds of metric are used,
depending on whether a field holds a single value or a list.

### 1.1 Single-value fields — accuracy

SROIE `vendor`, `date`, `total`, `address` and CORD `total` each hold one value
per document. For these, the metric is exact-match accuracy: the fraction of
documents whose predicted value matches the gold value, after field-appropriate
normalization (lowercase and punctuation-stripping for text, digits-only for
dates, a small numeric tolerance for totals).

For a field f over N documents:

    Accuracy(f) = (1 / N) * sum over documents of  1[ match(pred, gold) ]

where 1[.] is 1 when the prediction matches the gold value and 0 otherwise.

**Why accuracy is sufficient here.** With exactly one predicted value and one
gold value per field, a wrong prediction is at once a false positive and a false
negative, so FP = FN = N - TP. Substituting into the standard definitions makes
precision, recall, and F1 all reduce to the same value:

    Precision = TP / (TP + FP) = TP / N
    Recall    = TP / (TP + FN) = TP / N
    F1        = 2PR / (P + R)  = TP / N
    Accuracy  = TP / N

All four are identical for single-value fields, so reporting accuracy already
reports precision, recall, and F1. No additional columns are needed.

### 1.2 Line items — F1 (precision and recall differ)

A CORD receipt contains many line items, so the one-value equality above no
longer holds: a model can emit more items than gold (false positives) or fewer
(false negatives). Line items are therefore scored with F1 over matched items,
where an item matches when its description and price agree:

    Precision = TP / (TP + FP)      (of predicted items, how many are correct)
    Recall    = TP / (TP + FN)      (of gold items, how many were found)
    F1        = 2 * Precision * Recall / (Precision + Recall)

This is the only field where precision and recall carry separate information and
are worth reporting individually.

### 1.3 What is deliberately not used

- **Token-level F1 is not used.** These models output structured JSON fields,
  not raw page text, so field-level metrics are the correct measure. Token-level
  text overlap would not reflect extraction quality for structured output.
- **Latency** is recorded live per document (it cannot be recovered later).
  **Cost** for the local model is recorded as 0 at extraction time and computed
  afterwards as amortized GPU cost in the cost analysis.

### 1.4 Reading the tables

- Single-value fields are **accuracy**; line items are **F1**. The column
  headers should make this distinction explicit.
- A high line-item recall with lower precision means the model finds most items
  but also emits extras; the reverse means it is conservative but misses items.

---

## 2. Results

All arms are scored on the same test split: SROIE (347 documents) and CORD
(100 documents). Two fine-tuned variants are reported: `qwen_ft` (a single
adapter trained on both datasets) and `qwen_ft_sep` (separate per-dataset
adapters).

### 2.1 SROIE — field accuracy (347 documents)

| Field | Zero-shot | Fine-tuned (combined) | Fine-tuned (separate) |
|-------|-----------|-----------------------|-----------------------|
| vendor | 0.948 | 0.945 | 0.934 |
| date | 0.968 | 0.899 | 0.914 |
| total | 0.971 | 0.954 | 0.942 |
| address | 0.807 | 0.729 | 0.732 |

### 2.2 CORD — total accuracy and line-item F1 (100 documents)

| Metric | Zero-shot | Fine-tuned (combined) | Fine-tuned (separate) |
|--------|-----------|-----------------------|-----------------------|
| total (accuracy) | 0.885 | 0.969 | 0.969 |
| line_items (F1) | 0.389 | 0.806 | 0.814 |

### 2.3 What the results show

**Fine-tuning helps most where the task is structurally hard.** The clearest
effect is CORD line items, where F1 rises from 0.389 to about 0.81 — more than
doubling. Extracting and correctly structuring many line items per receipt is
exactly where the zero-shot model is weakest, and fine-tuning closes most of
that gap. CORD total also improves (0.885 to 0.969).

**Fine-tuning does not help, and slightly hurts, on already-strong simple
fields.** On SROIE, the zero-shot model is already above 0.94 on most fields.
After fine-tuning, the simple fields are flat or modestly lower (date 0.968 to
0.899; address 0.807 to 0.729). A likely explanation is that adapting to a small
training set trades some of the base model's broad robustness for fit to the
training distribution; where the base model was already near-saturated, there is
little to gain and some to lose.

**Combined and separate adapters perform almost identically.** `qwen_ft` and
`qwen_ft_sep` are within about one point on every field, with line-item F1 of
0.806 vs 0.814. Training a single combined adapter is therefore as effective as
maintaining a separate adapter per dataset, and is simpler and cheaper to serve.

**Summary.** The benefit of fine-tuning is concentrated where the zero-shot
model is weak (structured, multi-item extraction) rather than uniform across all
fields. This is a more useful conclusion than "fine-tuning always helps": it
indicates fine-tuning effort is best spent on structurally complex extraction,
while simple high-frequency fields are already well served zero-shot.

---

## 3. Why LoRA for fine-tuning

Fine-tuning here uses LoRA (Low-Rank Adaptation), specifically the 4-bit QLoRA
variant. The reasons are practical and methodological.

### 3.1 Hardware fit

The base model is Qwen2.5-VL-7B. Full fine-tuning updates all ~7B parameters and
requires keeping full-precision weights, gradients, and optimizer states in
memory — far beyond a single 16 GB GPU. LoRA freezes the base weights and trains
only small low-rank adapter matrices, so the number of trainable parameters
drops by orders of magnitude. Combined with 4-bit quantization of the frozen
base (QLoRA), the model both fine-tunes and runs inference within the available
VRAM budget.

### 3.2 Lower cost and faster iteration

Because only the adapter is trained, each run uses less memory and less compute,
and checkpoints are small (adapter-only, not a full model copy). This makes it
feasible to run several configurations and compare them, which is what produced
the combined-vs-separate adapter comparison above.

### 3.3 Preserves the base model

LoRA leaves the original weights untouched and adds a separable adapter. The
same base model can be reused with different adapters (for example, the separate
per-dataset adapters), and adapters can be swapped or removed without retraining.
This matches the experimental design, where several arms share one base model.

### 3.4 Comparison with the alternatives

- **Full fine-tuning** updates all weights and can reach the highest ceiling,
  but is memory- and compute-intensive, produces a full-size model per run, and
  carries a higher risk of overfitting on small training sets. It was not
  feasible on the available hardware and was not necessary for this comparison.
- **Prompt or prefix tuning** trains even fewer parameters than LoRA but is
  generally less expressive for changing extraction behaviour, and tends to
  underperform LoRA on structured tasks.
- **Zero-shot prompting** requires no training at all and is the baseline arm
  here; LoRA is the fine-tuned arm it is compared against.

LoRA sits at a practical middle point: enough capacity to materially change
behaviour (as the CORD line-item result shows), while remaining trainable and
serveable on a single mid-range GPU. For a study whose aim is a fair, repeatable
comparison rather than a maximum-accuracy single model, it is the appropriate
choice.

### 3.5 Fallback

If the primary fine-tuning script meets version or memory constraints, the
training data is also written in a sharegpt-style format consumable by ms-swift,
which can perform the same LoRA fine-tuning. This keeps the method reproducible
across tooling.

---

## 4. Reproducibility

- Predictions: `outputs/zeroshot__<dataset>__test.jsonl`,
  `outputs/qwen_ft__<dataset>__test.jsonl`,
  `outputs/qwen_ft_sep__<dataset>__test.jsonl`
- Scoring: `scripts/05_score.py` (reads the prediction files and compares each
  record's fields to the gold label; metrics can be changed without re-running
  extraction)
- Adapters: `outputs/qwen25vl_lora/` (combined), `outputs/sroie_lora/`,
  `outputs/cord_lora/` (separate)
- Same test split is used for every arm; the fine-tuning dev slice is taken from
  train only, never from test.
