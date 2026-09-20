"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it — the intent is that someone reviewing this file (a
stakeholder, a teammate, or future-you) can audit every decision without
having to guess at the reasoning.

FINAL FEATURE LIST is built at the bottom as FEATURE_COLUMNS.
"""

import pandas as pd


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # This dataset has zero missing values and zero duplicate rows
    # (verified in data_loader.py) — genuinely unusual for a real-world
    # dataset, and a direct consequence of it being simulated rather than
    # collected. Phase 5's normal "fix data quality issues" work has
    # nothing to do here; the honest data-quality note for this project
    # is the dataset's simulated origin itself (see data_loader.py),
    # not any missing-value or duplicate-row handling.
    # -----------------------------------------------------------------
    return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Engineered feature 1: high_price_ratio_flag
    # BUSINESS LOGIC: card issuers commonly use a purchase-size-relative-
    # to-history rule of thumb — a transaction several times larger than
    # a customer's typical purchase is a standard, real fraud-ops risk
    # trigger, independent of the raw ratio value's exact magnitude. This
    # gives the model an explicit, interpretable "this crossed a risk
    # threshold" signal on top of the continuous ratio_to_median_purchase_
    # price value it already has access to — not strictly necessary for
    # a tree-based model (which can learn its own threshold), but a
    # feature an analyst reviewing a flagged transaction can point to
    # directly, matching the same "explainability an agent can act on"
    # reasoning used for engineered features throughout this series.
    # WHY 3x: chosen as a round, defensible multiple of a customer's own
    # median purchase — an illustrative threshold, not one derived from
    # optimizing against this specific dataset.
    # -----------------------------------------------------------------
    df["high_price_ratio_flag"] = (df["ratio_to_median_purchase_price"] > 3).astype(int)

    # -----------------------------------------------------------------
    # Engineered feature 2: card_not_present_no_pin
    # BUSINESS LOGIC: an online order (card-not-present) combined with no
    # PIN verification is the classic highest-risk transaction TYPE in
    # card fraud — neither factor alone is as strong a signal as the
    # combination. EDA confirms both individually associate with fraud
    # (online_order: 94.6% of fraud vs. 62.2% of legitimate transactions;
    # used_pin_number: 0.3% of fraud vs. 11.0% of legitimate transactions)
    # — this feature lets the model use the INTERACTION directly rather
    # than having to discover the AND-condition from two separate columns
    # on its own.
    # -----------------------------------------------------------------
    df["card_not_present_no_pin"] = ((df["online_order"] == 1) & (df["used_pin_number"] == 0)).astype(int)

    # -----------------------------------------------------------------
    # Engineered feature 3: unusual_location_pattern
    # BUSINESS LOGIC: a transaction that is BOTH far from the customer's
    # home AND far from their last transaction location is a different,
    # stronger risk pattern than either distance being high in isolation
    # (e.g. someone who moved and is shopping near their new home has
    # high distance_from_home but LOW distance_from_last_transaction —
    # a normal pattern this feature correctly does NOT flag). Thresholds
    # are each column's own 90th percentile in the training data (not a
    # fixed number), since "far" is relative to this dataset's overall
    # distribution, not an absolute km value.
    # -----------------------------------------------------------------
    home_p90 = df["distance_from_home"].quantile(0.90)
    last_p90 = df["distance_from_last_transaction"].quantile(0.90)
    df["unusual_location_pattern"] = (
        (df["distance_from_home"] > home_p90) & (df["distance_from_last_transaction"] > last_p90)
    ).astype(int)

    return df


NUMERIC_MODEL_FEATURES = [
    "distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price",
    "repeat_retailer", "used_chip", "used_pin_number", "online_order",
    "high_price_ratio_flag", "card_not_present_no_pin", "unusual_location_pattern",
]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    return NUMERIC_MODEL_FEATURES


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print("\nFinal feature columns:")
    for c in feature_cols:
        print(f"  - {c}")
    print(f"\nhigh_price_ratio_flag rate: {engineered['high_price_ratio_flag'].mean():.3%}")
    print(f"card_not_present_no_pin rate: {engineered['card_not_present_no_pin'].mean():.3%}")
    print(f"unusual_location_pattern rate: {engineered['unusual_location_pattern'].mean():.3%}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
