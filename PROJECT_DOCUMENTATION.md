# Project Documentation: Credit Card Fraud Detection
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. It complements `README.md` (setup and run commands)
by explaining the architecture, data, decisions, and results in enough
depth that someone who did not build this project could maintain or extend
it. Every number below came from actually running the code in this
repository — none are illustrative, except where explicitly marked.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Score transactions in real time to flag likely fraud for review, within a fixed review-capacity budget |
| **Client context** | Card issuer / payments company's fraud operations team |
| **Data source** | Credit Card Fraud (Kaggle: dhanushnarayananr/credit-card-fraud) — **simulated**, not real transaction data |
| **Dataset size** | 1,000,000 transactions, 7 raw features |
| **Final model** | XGBoost Classifier, 10 features (7 raw + 3 engineered) |
| **Test recall @ 1% FPR** | 100% (see Section 5 for why this number needs a strong caveat, not a celebration) |
| **Business framing** | Catch fraud within a review team's realistic daily capacity to manually investigate flagged transactions |

---

## 2. Architecture

```
data/card_transdata.csv
        │
        ▼
data_loader.py ──────► loads raw CSV, documents simulated origin
        │
        ▼
clean_and_engineer.py ─► no cleaning needed (verified: 0 missing, 0
        │                duplicates); builds 3 business-logic features
        ▼
   split.py ───────────► stratified train/val/test split
        │
        ▼
train_model.py ───────► trains baseline (logistic regression),
        │                candidate (random forest), and final model
        │                (XGBoost); evaluates at a 1%-FPR operating
        │                point; saves artifacts
        ▼
outputs/fraud_model.joblib, feature_columns.joblib
        │
        ├──────────────► app.py ─── FastAPI service, /score endpoint,
        │                           reuses clean_and_engineer.py so
        │                           training and serving logic never
        │                           drift apart
        │
        └──────────────► monitor.py ─ PSI drift check + retrain trigger
                                       (same-snapshot stand-in — see
                                       Section 7 for why this dataset
                                       can't check genuine drift)
```

---

## 3. Data Dictionary (raw columns, as received)

| Column | Type | Description |
|---|---|---|
| `distance_from_home` | float | Distance (km) between the transaction location and the cardholder's home |
| `distance_from_last_transaction` | float | Distance (km) from the cardholder's previous transaction location |
| `ratio_to_median_purchase_price` | float | This transaction's amount relative to the cardholder's own historical median purchase |
| `repeat_retailer` | 0/1 | Whether this retailer has been used before |
| `used_chip` | 0/1 | Whether the transaction used chip verification |
| `used_pin_number` | 0/1 | Whether a PIN was used |
| `online_order` | 0/1 | Whether the transaction was online (card-not-present) |
| `fraud` | 0/1 | **Target** |

No cleaning was required — 0 missing values and 0 duplicate rows across
1,000,000 rows (verified in `data_loader.py`'s output), a direct
consequence of the dataset being simulated rather than collected from a
real payments system.

**Engineered features (not in the raw data):**

| Feature | Formula | Business rationale |
|---|---|---|
| `high_price_ratio_flag` | `ratio_to_median_purchase_price > 3` | A purchase several times a customer's typical spend is a standard fraud-ops risk trigger, given to the model as an explicit, analyst-readable threshold |
| `card_not_present_no_pin` | `online_order == 1 AND used_pin_number == 0` | The classic highest-risk transaction TYPE — the interaction, not either factor alone |
| `unusual_location_pattern` | Both distance features above their own 90th percentile | Distinguishes "genuinely unusual location" from a false positive like someone who just moved (high `distance_from_home`, low `distance_from_last_transaction`) |

Full reasoning for every decision above is inline in `clean_and_engineer.py`.

---

## 4. Key EDA Findings

From `eda.py`, run against the real dataset:

| Cut | Finding |
|---|---|
| Overall fraud rate | **8.74%** (much higher than real-world card fraud rates, typically <1% — a direct sign of simulated data) |
| Distance from home | Legitimate transactions average **22.8 km** from home; fraud averages **66.3 km** |
| Purchase ratio | Legitimate transactions average **1.42x** the customer's median purchase; fraud averages **6.01x** |
| PIN usage | **11.0%** of legitimate transactions use a PIN, vs. just **0.3%** of fraud — PIN verification is an extremely strong signal in this dataset |
| Online orders | **62.2%** of legitimate transactions are online, vs. **94.6%** of fraud |
| Repeat retailer | **88.2%** (legitimate) vs. **88.0%** (fraud) — a genuinely useful NEGATIVE finding: despite being an intuitively plausible risk signal, it barely differs between classes in this data and turned out to carry little weight in the final model (Section 5) |

A quick manual sanity check run during EDA: a 3-condition hand-written
rule (`ratio_to_median_purchase_price > 3 AND online_order AND NOT
used_pin_number`) alone achieves **70.5% recall / 66.6% precision** with
zero machine learning — a useful benchmark for how strong the raw
behavioral signal already is in this dataset before any modeling.

---

## 5. Modeling Results

### Experiment log (validation set)

| Model | Val PR-AUC | Val Recall @ 1% FPR |
|---|---|---|
| Logistic Regression (baseline) | 0.7478 | 0.3983 |
| Random Forest (30 trees, depth 10) | 1.0000 | 1.0000 |
| **XGBoost (final)** | 0.9997 | 1.0000 |

**This result needs an honest caveat, not a celebration.** Both
tree-based models hit essentially perfect separation. Combined with the
EDA finding that a simple 3-condition rule alone gets 70%+ recall, this
strongly suggests the dataset's fraud-generation process created cleanly
separable classes — a known property of simulated fraud data, not
evidence that fraud detection is this easy in the real world. Real
production fraud models rarely exceed 80-95% PR-AUC, because real
fraudsters actively adapt their behavior to evade whatever pattern the
current model is catching — an adversarial dynamic a static dataset like
this one cannot represent. **XGBoost was still selected as the final
model** (matching Random Forest's business metric while using
`tree_method="hist"` for a meaningfully faster train time — see Section
8) — but the takeaway to post about is the EVALUATION METHOD (recall at
a fixed review-capacity FPR, not raw accuracy), not the 100% number
itself.

### Held-out test set (final, unbiased evaluation)

| Metric | Value |
|---|---|
| Test PR-AUC | 0.9997 |
| Test recall @ 1% FPR | 100.00% |
| Decision threshold (score) | 0.0018 |
| N test transactions | 150,000 |
| N test fraud transactions | 13,110 |
| Legitimate transactions flagged for review | 1,369 (~0.99% of legitimate volume, confirming the 1% FPR budget was hit as intended) |

### What drives the model (feature importance)

Top 5 by importance:

1. `high_price_ratio_flag` (0.4440) — the engineered threshold feature, not the raw ratio
2. `ratio_to_median_purchase_price` (0.1721)
3. `card_not_present_no_pin` (0.1425) — the engineered interaction feature
4. `distance_from_home` (0.0688)
5. `repeat_retailer` (0.0536)

Notably, the two ENGINEERED features (`high_price_ratio_flag` and
`card_not_present_no_pin`) together account for **58.65%** of the
model's total decision weight — more than all 7 raw features combined.
This is a genuinely useful finding for the LinkedIn post: encoding
domain knowledge as explicit features didn't just aid explainability,
it measurably concentrated the model's decision-making on
business-interpretable signals rather than the model having to
rediscover the same patterns from raw numbers on its own.

### Business validation

On the 150,000-transaction held-out test set:

- The model caught an estimated **13,110 of 13,110** fraud transactions
  (100% recall) while flagging only **1,369** legitimate transactions
  for review (0.99% of legitimate volume — matching the 1% budget)
- **This dataset has no dollar-amount field at all.** Using a fully
  external, illustrative $500 average-fraud-amount figure (not derived
  from this dataset in any way — see the note in
  `business_validation.json`), the catch count implies an estimated
  **$6,555,000** in prevented fraud losses on this test set. This number
  should be treated as illustrating the CALCULATION METHOD a real
  business case would use, not as a number to act on — unlike the other
  projects in this series where at least part of the dollar estimate was
  anchored to real data, here none of it is.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser-based test form (every field wired) |
| `/docs` | GET | Auto-generated interactive API docs (Swagger UI) |
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/score` | POST | Score a single transaction — see request/response shape below |

**Response (verified against two contrasting real transactions):**
```json
{
  "fraud_risk_score": 1.0,
  "flagged_for_review": true,
  "top_factors": ["high_price_ratio_flag", "ratio_to_median_purchase_price", "card_not_present_no_pin"]
}
```
An ordinary transaction (near home, typical spend, in-person with PIN)
scored **0.0** on the same model — confirming the API responds to
genuinely different inputs, not a fixed value. `DECISION_THRESHOLD`
(0.0018) is the score cutoff calibrated to the 1% false-positive-rate
budget during training, not an arbitrary 0.5.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` implements, and was run to confirm works correctly:

- **Feature drift check** via PSI on `distance_from_home`,
  `ratio_to_median_purchase_price`, `online_order`, `used_pin_number`.
  All four came back at essentially zero PSI — expected, since this
  dataset has no time dimension and train/test come from the same
  stratified split of one static snapshot.
- **Performance decay check**: compares current recall@1%FPR against the
  baseline; triggers a retrain recommendation if it drops more than 0.05.
  On this run: no drop, so no retrain triggered.

**Important limitation stated directly in the script's output**: unlike
the sales-forecasting project in this series (where the test set is
genuinely future data), this dataset has no timestamp at all, so this
monitoring check can only confirm the CODE works — it cannot detect real
fraud-pattern drift, which in production would require timestamped
transactions and, ideally, tracking score-distribution shifts as
fraudsters adapt their tactics over time.

---

## 8. Known Limitations (stated for the handover record)

1. **This is a simulated dataset with an 8.74% fraud rate**, far higher
   than real-world card fraud rates (typically under 1%). The model's
   near-perfect performance is a property of this dataset's clean
   separability, not evidence of real-world fraud-detection difficulty
   — see Section 5 for the full discussion.
2. **No dollar-amount field exists**, so the entire business-impact
   dollar estimate is illustrative, more heavily caveated than the
   equivalent figures in the other projects in this series.
3. **No timestamp exists**, so genuine drift monitoring (fraud patterns
   changing over time, as real fraudsters adapt) cannot be evaluated
   here — only that the monitoring code itself functions correctly.
4. **Random Forest was trained at reduced size** (30 trees, depth 10)
   for single-core compute budget reasons, the same tradeoff made
   explicit in the sales-forecasting project.
5. **A production version needs real transaction data before
   deployment.** This walkthrough demonstrates the modeling, evaluation,
   and deployment METHOD — business-relevant metric selection
   (recall@fixed-FPR over raw accuracy), explainable feature
   engineering, and honest reporting of an implausibly clean result —
   which transfers to real data even though the specific numbers here
   would not.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document its simulated origin |
| `eda.py` | 6 | Fraud-vs-legitimate behavioral comparisons, summary chart |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Documents lack of cleaning needed, builds 3 engineered features — fully commented |
| `split.py` | 7 | Stratified train/val/test split |
| `train_model.py` | 8-9 | Model training, evaluation, business validation |
| `app.py` | 10 | FastAPI scoring service + fully-wired test form |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions, plus the honest "too easy" finding up front |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
