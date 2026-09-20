"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: a fraud-ops stakeholder's first question is
always "what does fraud actually LOOK like in this data" — not a general
statistical profile of every column. These cuts directly test the
behavioral patterns real card-fraud literature commonly cites (unusual
distance, card-not-present risk, purchase-size spikes, PIN verification)
against this specific dataset, rather than assuming they hold.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()

    print(f"Overall fraud rate: {df['fraud'].mean():.4%}")

    print("\n=== Mean feature value by fraud status ===")
    print(df.groupby("fraud").mean().round(3).T)
    # WHY: a single table answering "what's different about fraud
    # transactions on average" is the fastest way to sanity-check which
    # features are actually going to matter before any modeling.

    print("\n=== used_pin_number rate by fraud status ===")
    print(df.groupby("fraud")["used_pin_number"].mean().round(4))
    # WHY: tests the "PIN verification prevents fraud" hypothesis
    # directly — a claim worth checking, not assuming.

    print("\n=== online_order rate by fraud status ===")
    print(df.groupby("fraud")["online_order"].mean().round(4))
    # WHY: tests whether card-not-present (online) transactions are
    # genuinely riskier in THIS data, the standard industry assumption.

    print("\n=== repeat_retailer rate by fraud status ===")
    print(df.groupby("fraud")["repeat_retailer"].mean().round(4))
    # WHY: a genuinely useful NEGATIVE finding worth checking explicitly
    # — unlike the features above, this one turns out to barely differ
    # between fraud and non-fraud (see PROJECT_DOCUMENTATION.md Section 4)
    # despite being intuitively plausible as a risk signal.

    print(f"\n=== Data quality ===")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # ---- Chart: the three strongest behavioral splits ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    df.groupby("fraud")["distance_from_home"].mean().plot(
        kind="bar", ax=axes[0], color="#4C72B0", title="Avg distance from home"
    )
    axes[0].set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    axes[0].set_ylabel("Distance (km)")

    df.groupby("fraud")["ratio_to_median_purchase_price"].mean().plot(
        kind="bar", ax=axes[1], color="#C44E52", title="Avg ratio to median purchase price"
    )
    axes[1].set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    axes[1].set_ylabel("Ratio")

    pin_rate = df.groupby("fraud")["used_pin_number"].mean()
    pin_rate.plot(kind="bar", ax=axes[2], color="#55A868", title="PIN usage rate")
    axes[2].set_xticklabels(["Legitimate", "Fraud"], rotation=0)
    axes[2].set_ylabel("Share using PIN")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
