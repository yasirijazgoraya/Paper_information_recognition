# Evaluation Metrics

Formal definitions of every metric reported in `RESULTS.md`. All scorers are implemented in `code/layer1_vision/score_v3.py`.

## Notation

For a document $d$:
- $P_d$ — multiset of tokens produced by the OCR backend.
- $G_d$ — multiset of ground-truth tokens.
- $T(\cdot)$ — tokenizer: lowercase, strip punctuation, split on whitespace.
- $\text{norm}(\cdot)$ — string normalization (lowercase, strip whitespace, remove punctuation; CORD additionally removes the Indonesian thousands separator).
- $N$ — number of documents in the test split.
- $\mathbb{1}[\cdot]$ — indicator function (1 if true, else 0).

---

## 1. Token-level Precision, Recall, F1

Per document, treating tokens as a multiset:

$$
\text{Precision}_d \;=\; \frac{|P_d \cap G_d|}{|P_d|},
\qquad
\text{Recall}_d \;=\; \frac{|P_d \cap G_d|}{|G_d|},
\qquad
F_1^{(d)} \;=\; \frac{2\,\text{Precision}_d\,\text{Recall}_d}{\text{Precision}_d + \text{Recall}_d}.
$$

Macro-average across the test split:

$$
\text{F1} \;=\; \frac{1}{N}\sum_{d=1}^{N} F_1^{(d)}.
$$

**What it captures:** whether the OCR recovered the right *vocabulary* of the page. Ignores order, position, and structure. Useful as a coarse quality signal; insufficient for structured extraction.

---

## 2. SROIE field hit rate

For each target field $f \in \{\text{company},\,\text{date},\,\text{address},\,\text{total}\}$, with predicted value $\hat{y}_f^{(d)}$ and ground-truth $y_f^{(d)}$:

$$
\text{hit}_f^{(d)} \;=\;
\begin{cases}
\mathbb{1}\!\left[\hat{y}_f^{(d)} = y_f^{(d)}\right] & \text{if } f \in \{\text{date},\,\text{total}\} \\[6pt]
\mathbb{1}\!\left[\text{norm}(\hat{y}_f^{(d)}) = \text{norm}(y_f^{(d)})\right] & \text{if } f = \text{company} \\[6pt]
\mathbb{1}\!\left[\,J\!\left(T(\hat{y}_f^{(d)}),\,T(y_f^{(d)})\right) \ge 0.7\,\right] & \text{if } f = \text{address (Fix B)}
\end{cases}
$$

where the token-set overlap (Jaccard index) is

$$
J(A, B) \;=\; \frac{|A \cap B|}{|A \cup B|}.
$$

Per-field hit rate over the split:

$$
\text{HitRate}_f \;=\; \frac{1}{N}\sum_{d=1}^{N} \text{hit}_f^{(d)}.
$$

Macro field recall (the *Field / Key Recall* column in the headline table):

$$
\text{FieldRecall} \;=\; \frac{1}{|\mathcal{F}|}\sum_{f \in \mathcal{F}} \text{HitRate}_f.
$$

**What it captures:** the operational quality of receipt extraction — did the system actually recover each labelled field, allowing for minor surface differences on multi-line addresses.

---

## 3. CORD numeric normalization (Fix A) and field recall

CORD prices use periods as thousands separators (Indonesian convention). Without normalization, `25.000` is read as 25.0 instead of 25 000.

Let $\rho$ be the regular expression `^\d{1,3}(\.\d{3})+$` (one to three digits, then one or more groups of `.` followed by three digits). Define:

$$
\nu(t) \;=\;
\begin{cases}
\text{strip\_periods}(t) & \text{if } t \text{ matches } \rho \\
t & \text{otherwise.}
\end{cases}
$$

For each ground-truth field instance $g \in G_d^{(f)}$ in category $f \in \{\text{menu items},\,\text{menu prices},\,\text{totals}\}$:

$$
\text{FieldRecall}_f \;=\; \frac{1}{N}\sum_{d=1}^{N}\;\frac{\big|\{\,g \in G_d^{(f)} \;:\; \nu(g) \in \nu(P_d)\,\}\big|}{|G_d^{(f)}|}.
$$

**What it captures:** of all the items / prices / totals on a real receipt, what fraction did the OCR find? Numeric strings are compared after Fix A so format differences do not count as misses.

---

## 4. FUNSD answer-token recall (lenient)

Let $A_d \subseteq G_d$ be the subset of ground-truth tokens labelled as part of an *answer* span. Lenient recall ignores question text and structural pairing:

$$
\text{AnsRecall}_d \;=\; \frac{|A_d \cap P_d|}{|A_d|},
\qquad
\text{AnsRecall} \;=\; \frac{1}{N}\sum_{d=1}^{N} \text{AnsRecall}_d.
$$

**What it captures:** did the OCR read the answer values anywhere on the page?

---

## 5. FUNSD Q → A pair recall (strict, Fix C)

Let $\mathcal{Q}_d$ be the set of $(q, a)$ pairs in document $d$, where $q$ and $a$ are token sequences. A pair is *hit* only if **both** its question tokens and its answer tokens appear in the predicted output:

$$
\text{hit}(q, a, P_d) \;=\; \mathbb{1}\!\left[\,T(q) \subseteq P_d \;\land\; T(a) \subseteq P_d\,\right].
$$

$$
\text{PairRecall}_d \;=\; \frac{1}{|\mathcal{Q}_d|}\sum_{(q,a)\,\in\,\mathcal{Q}_d}\text{hit}(q, a, P_d),
\qquad
\text{PairRecall} \;=\; \frac{1}{N}\sum_{d=1}^{N}\text{PairRecall}_d.
$$

**What it captures:** the OCR must surface the question *and* its answer together, not just one or the other. This is the metric that exposes how much downstream layout / relation modelling is still needed.

---

## 6. Latency

Wall-clock time per document on the local GPU, averaged over the split:

$$
\bar{\ell} \;=\; \frac{1}{N}\sum_{d=1}^{N} \ell_d,
$$

with $\ell_d$ measured from image-load to final JSON emission, excluding one-off model-load time (amortized via a warm-up pass).

**What it captures:** throughput on production hardware. Compared head-to-head because all three backends ran on the same GPU and same image set.

---

## 7. What these metrics do **not** capture

- **Confidence intervals.** All numbers in `RESULTS.md` are point estimates. A bootstrap 95 % CI on F1 is roughly $\pm 0.05$ for $N = 50$ (FUNSD) and $\pm 0.02$ for $N = 347$ (SROIE).
- **Precision on CORD.** Section 3 reports recall only; a model that hallucinates extra menu items would not be penalized here. Future work: add a symmetric F1.
- **Field-level F1 on SROIE.** Section 2 reports hit rate (binary per field). Equivalent to per-field precision = recall when the system emits exactly one value per field, but breaks if multiple candidates are emitted.
- **OCR detection IoU.** Bounding-box quality is not scored — only text content. Adequate for downstream KIE, insufficient for layout-aware tasks.
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
