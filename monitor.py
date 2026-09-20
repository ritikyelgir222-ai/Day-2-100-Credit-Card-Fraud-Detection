"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index) specifically: PSI is the standard,
interview-recognizable metric for feature drift in industry because it
gives a single interpretable number per feature with well-established
severity thresholds (< 0.1 stable, 0.1-0.25 moderate shift, > 0.25
significant shift needing action).

WHY THIS SCRIPT SIMULATES A "NEW BATCH" instead of using genuinely new
data: this dataset has no timestamp column at all (it's a simulated,
cross-sectional set of transactions — see data_loader.py), so unlike the
sales-forecasting project there's no real "future" data to compare
against even in principle. We reuse the held-out test set as a stand-in,
which mainly confirms the monitoring code path works correctly, not that
real-world fraud-pattern drift is being detected — a real production
system's monitoring would need timestamped transaction data this dataset
doesn't provide (see PROJECT_DOCUMENTATION.md Section 8).
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import recall_at_fpr, TARGET_FPR

RECALL_DROP_THRESHOLD = 0.05  # per SDLC doc Phase 13: retrain if recall@1%FPR drops more than this
DRIFT_FEATURES = ["distance_from_home", "ratio_to_median_purchase_price", "online_order", "used_pin_number"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/fraud_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    X = engineered[feature_cols]
    y = engineered["fraud"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("=== Feature Drift (PSI): train distribution vs. held-out batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY NO DRIFT IS EXPECTED HERE: train/test came from the same
    # stratified split of one static, simulated dataset with no time
    # axis, so we EXPECT near-zero PSI — this run confirms the
    # monitoring code itself works, not that real fraud-pattern drift is
    # being caught (unlike the sales-forecasting project, where the test
    # set genuinely is future data).

    scores = model.predict_proba(X_test)[:, 1]
    recall, threshold = recall_at_fpr(y_test, scores)

    print(f"\n=== Performance on held-out batch ===")
    print(f"Recall at {TARGET_FPR:.0%} FPR: {recall:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_recall = json.load(f)["test_recall_at_1pct_fpr"]

    drop = baseline_recall - recall
    if drop > RECALL_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: recall dropped by {drop:.3f} (threshold: {RECALL_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (recall change: {drop:.3f}, threshold: {RECALL_DROP_THRESHOLD})")

    print("\nNOTE: a real fraud-monitoring system would also track SCORE")
    print("DISTRIBUTION drift over time (fraudsters adapt their tactics),")
    print("which requires timestamped transactions this dataset doesn't have.")


if __name__ == "__main__":
    run_monitoring_check()
