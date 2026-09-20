# Credit Card Fraud Detection — Full Project Code (Real, Simulated Dataset)

Companion code to Day 2 of the 100-day LinkedIn series. Uses Kaggle's
"Credit Card Fraud" dataset, run end to end — every number in
`PROJECT_DOCUMENTATION.md` came from executing this code, not from an
illustrative example. **Important:** this dataset is simulated, not real
transaction data — see the honesty note below before treating any result
here as "solved."

## Data source

**Credit Card Fraud** (Kaggle: dhanushnarayananr/credit-card-fraud) —
1,000,000 simulated transactions, 7 behavioral features (distance from
home, distance from last transaction, purchase-price ratio, repeat
retailer, chip/PIN usage, online order) plus a binary fraud label.

The CSV is already included at `data/card_transdata.csv`. If you'd
rather pull it fresh from Kaggle: [the dataset
page](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
— the schema is identical.

## Read this before posting: the honest, important finding

The final model hit **100% recall at a 1% false-positive rate** on the
held-out test set. That is NOT a result to post as "I built a perfect
fraud detector" — it's a sign this simulated dataset is far more
separable than real fraud data ever is. A quick sanity check in
`train_model.py`'s development (a 3-condition hand-written rule: high
price ratio AND online AND no PIN) already gets 70.5% recall / 66.6%
precision on its own — this dataset's fraud pattern is strong and
clean by design, not because fraud is actually this easy to catch in the
real world. Real fraud detection rarely exceeds 80-95% PR-AUC because
real fraudsters actively adapt to evade whatever the current model
flags — an adversarial dynamic this static, simulated dataset doesn't
have. **The honest post here is about the METHOD** (business-relevant
metric selection, explainable feature engineering, deployment
discipline) — not about the specific accuracy number, which would be
misleading to present as representative of real-world fraud detection
difficulty.

## Why every script is commented the way it is

- **`recall_at_fpr`, not accuracy or plain recall, is the business
  metric** — a fraud review team has a fixed daily review capacity;
  capping false positives at 1% directly encodes that constraint (see
  `train_model.py`).
- **Three engineered features encode real fraud-ops risk logic**:
  `high_price_ratio_flag` (a purchase several times a customer's typical
  spend), `card_not_present_no_pin` (the classic highest-risk
  combination), and `unusual_location_pattern` (far from home AND far
  from the last transaction simultaneously — not either alone, which
  would also flag someone who just moved).
- **The business-impact dollar figure is caveated more strongly than in
  the rest of this series** — this dataset has no dollar-amount column
  at all, so unlike the churn or house-price projects (which had at
  least one real number to anchor to), the entire fraud-loss estimate
  here is illustrative. Stated explicitly, not glossed over.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & inspects the raw CSV
python eda.py                  # Phase 6 — fraud-vs-legitimate behavioral splits + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — feature engineering (dataset needs no cleaning — see the file)
python train_model.py          # Phase 8-9 — baseline + final model, business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "distance_from_home": 80, "distance_from_last_transaction": 45,
        "ratio_to_median_purchase_price": 8.5, "repeat_retailer": false,
        "used_chip": false, "used_pin_number": false, "online_order": true
      }'
```

Expected: a risk score of ~1.0 and `"flagged_for_review": true` — this
profile (far from home, large spend spike, online, no PIN) matches
exactly the pattern EDA identifies as highest-risk.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the dataset, documents its simulated origin honestly |
| `eda.py` | 6 | Fraud-vs-legitimate behavioral comparisons, summary chart |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Documents the lack of cleaning needed, builds 3 business-logic engineered features |
| `split.py` | 7 | Stratified train/val/test split, with reasoning |
| `train_model.py` | 8-9 | Baseline (logistic regression) → Random Forest → XGBoost, recall-at-1%-FPR business evaluation |
| `app.py` | 10 | FastAPI scoring service — every form field wired, none hardcoded |
| `monitor.py` | 13 | PSI-based drift check + recall-drop retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`
- `eda_summary.png`
- `experiment_log.csv` — baseline vs. candidate models
- `business_validation.json` — recall@1%FPR, illustrative dollar impact (see the note field)
- `feature_importance.csv`
- `fraud_model.joblib`, `feature_columns.joblib`
- `model_card.json` — includes explicit known limitations

## Known limitations (stated honestly, not hidden)

- **This is a simulated dataset with an 8.74% fraud rate**, far higher
  than real-world card fraud (typically under 1%). The near-perfect
  model performance reflects this dataset's clean separability, not
  real-world fraud-detection difficulty.
- **No dollar-amount field exists in this dataset**, so the business
  case's dollar figure is fully illustrative.
- **No timestamp exists**, so `monitor.py` can't check for genuine
  fraud-pattern drift over time — only that the monitoring code itself
  runs correctly.
- **Random Forest was trained at reduced size** (30 trees, depth 10) for
  single-core compute budget reasons.
