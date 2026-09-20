"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that scores a single transaction using the
fields a payment processor's authorization system would have at
transaction time. The cleaning and feature-engineering logic from
clean_and_engineer.py is reused here rather than duplicated, so a
transformation change only ever needs to happen in one place — this
guards against training/serving skew.

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Transaction Fraud Risk Scoring API", version="1.0")

MODEL = joblib.load("outputs/fraud_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")
DECISION_THRESHOLD = 0.0018  # from outputs/model_card.json — calibrated to a 1% false-positive-rate budget


class TransactionRecord(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: bool
    used_chip: bool
    used_pin_number: bool
    online_order: bool


class FraudScoreResponse(BaseModel):
    fraud_risk_score: float
    flagged_for_review: bool
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Transaction Fraud Scoring</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Transaction Fraud Scoring — Test Form</h2>
        <p>Every field below maps 1:1 to a field in <code>TransactionRecord</code>
           (app.py) — nothing here is hardcoded in the page itself. Full API
           docs at <a href="/docs">/docs</a>.</p>
        <form id="scoreForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Distance from home (km)<br><input name="distance_from_home" type="number" step="0.01" value="15" required></label>
            <label>Distance from last transaction (km)<br><input name="distance_from_last_transaction" type="number" step="0.01" value="2" required></label>
            <label>Ratio to median purchase price<br><input name="ratio_to_median_purchase_price" type="number" step="0.01" value="1.2" required></label>
            <label>Repeat retailer?<br>
                <select name="repeat_retailer"><option value="true" selected>Yes</option><option value="false">No</option></select>
            </label>
            <label>Used chip?<br>
                <select name="used_chip"><option value="true" selected>Yes</option><option value="false">No</option></select>
            </label>
            <label>Used PIN?<br>
                <select name="used_pin_number"><option value="true" selected>Yes</option><option value="false">No</option></select>
            </label>
            <label>Online order?<br>
                <select name="online_order"><option value="false" selected>No</option><option value="true">Yes</option></select>
            </label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Score this transaction</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("scoreForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const asFloat = (name) => parseFloat(form.get(name));
            const asBool = (name) => form.get(name) === "true";
            const payload = {
                distance_from_home: asFloat("distance_from_home"),
                distance_from_last_transaction: asFloat("distance_from_last_transaction"),
                ratio_to_median_purchase_price: asFloat("ratio_to_median_purchase_price"),
                repeat_retailer: asBool("repeat_retailer"),
                used_chip: asBool("used_chip"),
                used_pin_number: asBool("used_pin_number"),
                online_order: asBool("online_order")
            };
            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                document.getElementById("result").innerText =
                    "Error " + res.status + ": " + await res.text();
                return;
            }
            const data = await res.json();
            document.getElementById("result").innerText =
                "Risk score: " + data.fraud_risk_score +
                " | Flagged: " + data.flagged_for_review +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """


@app.post("/score", response_model=FraudScoreResponse)
def score_transaction(record: TransactionRecord):
    raw = record.model_dump()
    raw_row = {
        "distance_from_home": raw["distance_from_home"],
        "distance_from_last_transaction": raw["distance_from_last_transaction"],
        "ratio_to_median_purchase_price": raw["ratio_to_median_purchase_price"],
        "repeat_retailer": 1 if raw["repeat_retailer"] else 0,
        "used_chip": 1 if raw["used_chip"] else 0,
        "used_pin_number": 1 if raw["used_pin_number"] else 0,
        "online_order": 1 if raw["online_order"] else 0,
    }
    raw_df = pd.DataFrame([raw_row])

    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba = float(MODEL.predict_proba(X)[0, 1])
    flagged = proba >= DECISION_THRESHOLD

    importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return FraudScoreResponse(
        fraud_risk_score=round(proba, 4),
        flagged_for_review=flagged,
        top_factors=top_factors,
    )
