"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "Credit Card Fraud" dataset published on Kaggle by
user dhanushnarayananr:
    https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

It contains 1,000,000 simulated card transactions with 7 behavioral/
contextual features plus a binary fraud label. Per the dataset's own
documentation, it was "sourced by some unnamed institute" — i.e. this is
a realistically-structured but SIMULATED dataset, not a released real
transaction log (real fraud data is essentially never released publicly
for confidentiality reasons — see WHY THIS DATASET below for why that
still makes it a reasonable choice for this walkthrough).

If you're following along on Kaggle: download `card_transdata.csv` from
the link above and place it at `data/card_transdata.csv` — the schema is
identical to the file already included in this project.

WHY THIS DATASET (AND ITS HONEST LIMITS)
-----------------------------------------
- Its features are interpretable (distance from home, chip/PIN usage,
  online vs. in-person) rather than the anonymized PCA components (V1-V28)
  in the other commonly-used Kaggle credit card fraud dataset — this
  matters a lot for THIS series specifically, since interpretable
  features are what make a genuine business-perspective EDA possible at
  all. PCA-anonymized features can build an accurate model, but they
  can't support a "here's WHY this looks like fraud" narrative.
- BEING SIMULATED IS A REAL LIMITATION, STATED UP FRONT: an 8.74% fraud
  rate is far higher than real-world card fraud rates (typically well
  under 1%), and the feature relationships are likely cleaner/stronger
  than messy real transaction data would be. Treat this project as
  demonstrating the METHOD — the modeling, evaluation, and deployment
  approach — not as literal findings a real fraud team should deploy
  unchanged. See PROJECT_DOCUMENTATION.md Section 8 for the full list of
  limitations this creates.
"""

import pandas as pd

RAW_DATA_PATH = "data/card_transdata.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Fraud rate: {df['fraud'].mean():.4%}")
    print(f"\nMissing values: {df.isnull().sum().sum()}  |  Duplicate rows: {df.duplicated().sum()}")
    print("\nColumn dtypes:")
    print(df.dtypes)
