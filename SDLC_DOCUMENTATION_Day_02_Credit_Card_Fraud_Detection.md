# SDLC Documentation: Credit Card Fraud Detection
### Payments / FinTech Industry | Machine Learning | Day 2 of the 100-Day Series
### Filled against the same 14-phase SDLC structure used for Day 1

> **Data honesty note:** This project uses a simulated Kaggle transaction dataset. The reported model metrics are valid for this dataset, but they should not be interpreted as evidence that real-world credit-card fraud is equally easy to detect.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Business problem owner:** Fraud Operations / Payments Risk team
- **Day-to-day users:** Fraud analysts and transaction-review teams
- **Business problem:** Automatically identify suspicious transactions while keeping the number of legitimate transactions sent for manual review within a fixed operational capacity.
- **Decision this project informs:** Which transactions should be flagged for fraud review.
- **Operational constraint:** Fraud detection cannot be evaluated only by accuracy. A fraud team has limited review capacity, so the model must achieve strong fraud recall at a controlled false-positive rate.
- **Primary decision metric:** Recall at a fixed **1% False Positive Rate (FPR)**.
- **Important data constraint:** The dataset is simulated and does not contain transaction timestamps or transaction dollar amounts.
- **Deliverable — Problem Statement:**
  > "Build a transaction-level fraud scoring system that captures as much fraud as possible while limiting legitimate transactions sent to manual review to approximately 1%."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** Maximize fraud detection within a fixed legitimate-transaction review capacity.
- **Scope:**
  - In-scope: 1,000,000 simulated card transactions.
  - Seven raw behavioral transaction features plus engineered fraud-risk features.
  - Transaction-level fraud probability/scoring.
- **Out-of-scope:**
  - Real payment-network integration.
  - Financial loss estimation using actual transaction amounts.
  - Time-series fraud adaptation.
  - Production fraud investigation workflow.
- **Success metrics / KPIs:**
  - Primary: Recall @ 1% FPR.
  - Secondary: PR-AUC.
  - Operational metric: number of legitimate transactions flagged at the selected threshold.
- **Business threshold:** A threshold of **0.0018** was calibrated to operate at approximately 1% FPR on the held-out test evaluation.
- **Assumptions & constraints:**
  - The dataset is simulated.
  - Fraud prevalence is **8.74%**, which is much higher than typically expected in real payment environments.
  - No transaction amount is available.
  - No timestamp is available.
  - No measured financial-loss or fraud-review outcome data is available.
- **Sign-off:** N/A — self-directed portfolio project. A real production deployment would require approval from Fraud Operations, Risk, Compliance, and relevant engineering stakeholders.

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / ML Engineer | **Output:** FRD/TRD

### Functional requirements

- Load transaction data from the approved source.
- Validate data quality before modeling.
- Engineer business-driven fraud indicators.
- Train and compare multiple classification models.
- Select a threshold based on the operational 1% FPR requirement.
- Return a fraud score and fraud flag for an individual transaction.
- Provide the major risk factors associated with the scoring logic.
- Expose the scoring system through a REST API.

### Non-functional requirements

- Training and serving must use the same feature-engineering logic.
- The API must support a single transaction scoring request.
- The model and feature-column list must be persisted as reusable artifacts.
- Monitoring must provide a repeatable drift/retraining-check mechanism.

### Data source

- Kaggle simulated credit-card transaction dataset.
- Dataset size: **1,000,000 transactions**.
- Raw behavioral columns:
  - `distance_from_home`
  - `distance_from_last_transaction`
  - `ratio_to_median_purchase_price`
  - `repeat_retailer`
  - `used_chip`
  - `used_pin_number`
  - `online_order`
  - `fraud`

---

## Phase 4: Project Planning
**Owner:** Project / Delivery Manager | **Output:** Project plan, risk register

### Build sequence

1. Data loading and validation
2. Exploratory data analysis
3. Cleaning and business feature engineering
4. Stratified train/validation/test split
5. Baseline and candidate model training
6. Fixed-FPR model evaluation
7. Final model selection
8. Feature-importance analysis
9. FastAPI deployment
10. Monitoring and retraining-trigger implementation
11. Documentation and model-card preparation

### Risk register

- **Risk:** Simulated data may produce unrealistically strong model performance.
  - **Mitigation:** Explicitly document the simulated nature of the dataset and avoid presenting the result as production fraud performance.
- **Risk:** High fraud prevalence may not represent real payment traffic.
  - **Mitigation:** Report the 8.74% fraud rate and state that production performance must be validated on real transaction data.
- **Risk:** No timestamp prevents genuine temporal drift analysis.
  - **Mitigation:** Clearly label PSI monitoring as a same-snapshot simulation.
- **Risk:** No transaction amount prevents real dollar-loss validation.
  - **Mitigation:** Do not present the illustrative $500 fraud-loss calculation as actual business impact.
- **Risk:** Feature-engineering differences between training and API serving could create model-serving skew.
  - **Mitigation:** Reuse the same `clean_and_engineer.py` logic in both pipelines.

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

### Data inventory

- Source: Kaggle simulated credit-card fraud dataset.
- Rows: **1,000,000**
- Raw behavioral features: **7**
- Target: `fraud`

### Data quality findings

- No missing values were identified.
- No duplicate transactions were identified.
- The absence of missing values and duplicates is consistent with the simulated nature of the dataset.
- Fraud prevalence: **8.74%**.

### Raw behavioral fields

| Feature | Business meaning |
|---|---|
| `distance_from_home` | Distance of transaction from customer's home location |
| `distance_from_last_transaction` | Distance from the customer's previous transaction |
| `ratio_to_median_purchase_price` | Purchase value relative to the customer's median purchase price |
| `repeat_retailer` | Whether the retailer has been used previously |
| `used_chip` | Whether chip payment was used |
| `used_pin_number` | Whether a PIN was used |
| `online_order` | Whether the transaction was online |
| `fraud` | Binary fraud target |

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** EDA report / summary

### Actual findings

- Overall fraud rate: **8.74%**
- Average distance from home:
  - Legitimate: **22.8 km**
  - Fraud: **66.3 km**
- Purchase-price ratio:
  - Legitimate: **1.42x**
  - Fraud: **6.01x**
- PIN usage:
  - Legitimate: **11.0%**
  - Fraud: **0.3%**
- Online orders:
  - Legitimate: **62.2%**
  - Fraud: **94.6%**
- Repeat retailer:
  - Legitimate: **88.2%**
  - Fraud: **88.0%**
  - This was a useful negative finding: repeat retailer behavior showed little separation between the two classes.

### Rule-based baseline

Before machine learning, a simple fraud-operations rule was evaluated:

> `ratio_to_median_purchase_price > 3` AND `online_order = 1` AND `used_pin_number = 0`

This rule achieved:

- Recall: **70.5%**
- Precision: **66.6%**

This established that the dataset already contains strong, human-interpretable fraud patterns before applying ML.

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`

### Cleaning

- No imputation or duplicate removal was required.
- The preprocessing pipeline preserves the seven raw behavioral features.

### Engineered features

Three business-oriented features were added:

1. **`high_price_ratio_flag`**
   - Flags transactions where the purchase-price ratio is greater than 3.
   - Encodes an unusually high purchase relative to the customer's normal purchase level.

2. **`card_not_present_no_pin`**
   - `online_order = 1` AND `used_pin_number = 0`
   - Encodes a card-not-present transaction without PIN authentication.

3. **`unusual_location_pattern`**
   - Flags transactions where both distance features are above their respective 90th percentiles.
   - Encodes an unusual geographic transaction pattern.

### Final feature set

- 7 raw behavioral features
- 3 engineered features
- **10 model features in total**

### Split strategy

- Stratified train/validation/test split.
- Stratification preserves the fraud/non-fraud class ratio across evaluation sets.
- A time-based split was not possible because the source dataset contains no transaction timestamp.

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, experiment log

### Candidate models

The following models were compared under the same operational constraint:

- Logistic Regression
- Random Forest
- XGBoost

### Validation results

| Model | PR-AUC | Recall @ 1% FPR |
|---|---:|---:|
| Logistic Regression | 0.7478 | 0.3983 |
| Random Forest | 1.0000 | 1.0000 |
| XGBoost | 0.9997 | 1.0000 |

### Model selection

XGBoost was carried forward as the final model.

The selection was based on the overall project design and production-oriented considerations, including the strong fixed-FPR performance and the suitability of XGBoost for the feature set.

### Important interpretation

The near-perfect results must be interpreted in the context of the simulated dataset. They are **not** evidence that real-world fraud detection will achieve 100% recall at 1% FPR.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist | **Output:** `business_validation.json`, feature-importance artifacts

### Held-out test set

- Test transactions: **150,000**
- Fraud transactions: **13,110**
- Legitimate transactions flagged: **1,369**

### Technical performance

- PR-AUC: **0.9997**
- Recall @ 1% FPR: **100.00%**
- Selected threshold: **0.0018**
- Approximate legitimate transactions flagged: **0.99% of legitimate volume**

### Business interpretation

At the selected operating point, the model captured:

- **13,110 / 13,110 fraud transactions**
- While flagging **1,369 legitimate transactions**

This directly connects the model evaluation to the fraud-review capacity constraint.

### Illustrative financial calculation

The dataset contains no transaction-amount field, so actual prevented-loss dollars cannot be calculated.

An external assumption of **$500 average fraud amount** would produce:

> 13,110 × $500 = **$6,555,000**

However, this is **illustrative only** and must not be presented as actual business impact from the model.

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

### Deployment architecture

```text
Transaction Input
       |
       v
FastAPI /score
       |
       v
Shared Feature Engineering
(clean_and_engineer.py)
       |
       v
XGBoost Model
       |
       +----> Fraud Score
       |
       +----> Fraud Flag
       |
       +----> Risk Factors
```

### API endpoints

- `/` — browser-based test interface
- `/docs` — FastAPI interactive API documentation
- `/health` — service health check
- `/score` — transaction fraud scoring endpoint

### Deployment discipline

The API reuses the same feature-engineering logic used during training. This reduces the risk of training/serving skew.

### Persisted artifacts

- `fraud_model.joblib`
- `feature_columns.joblib`
- `model_card.json`

### Production gaps

Not implemented in this portfolio version:

- Docker/containerization
- CI/CD pipeline
- Cloud deployment
- Authentication/authorization
- Real payment-network integration
- Production feature store

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** Test results

### Tests performed

- End-to-end training pipeline execution.
- Data-loading and feature-engineering validation.
- Model artifact generation.
- API scoring validation.
- High-risk and normal transaction scoring checks.
- `/health` endpoint verification.
- `/docs` API documentation availability.
- Verification that API scoring uses the same feature-engineering code as training.

### Expected scoring behavior

A high-risk transaction profile should produce a high fraud score and be flagged according to the calibrated threshold.

A normal transaction profile should produce a low score and remain below the fraud threshold.

### What was not performed

- Formal automated unit-test suite.
- Full integration testing with a real payment system.
- User acceptance testing with a fraud-operations team.
- Production load testing.
- Long-term temporal validation.

These are documented gaps rather than represented as completed production controls.

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** README, project documentation, model card, SDLC document

### Documentation delivered

- `README.md`
- `PROJECT_DOCUMENTATION.md`
- SDLC documentation
- Model card
- API documentation through FastAPI `/docs`
- Experiment and feature-importance outputs

### Handover information

The project documents:

- Business problem
- Dataset and data limitations
- EDA findings
- Feature engineering
- Model experiments
- Business metric selection
- Held-out test results
- API architecture
- Monitoring approach
- Production limitations

### Key handover warning

The most important handover note is that the model's exceptional performance is specific to the simulated dataset and must not be transferred directly to a production fraud environment.

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

### Drift monitoring

Population Stability Index (PSI) monitoring was implemented for:

- `distance_from_home`
- `distance_from_last_transaction`
- `ratio_to_median_purchase_price`
- `online_order`

### Monitoring result

The monitoring implementation operates on the available static dataset.

Because the dataset contains no timestamp and no genuinely new production batch, the observed PSI result should be treated as a **monitoring simulation**, not proof of production stability.

### Retraining trigger

A performance-based retraining condition is implemented:

- Monitor Recall @ 1% FPR.
- Trigger retraining when recall falls by more than **0.05** from the stored baseline.

### Production maintenance requirements

A real deployment should additionally monitor:

- Fraud prevalence
- False-positive rate
- Recall at operational review capacity
- Precision
- PR-AUC
- Feature drift
- Concept drift
- Data-quality failures
- API latency and error rate
- Fraud patterns over time

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed portfolio project) | **Output:** Complete project package

### Closure against the business objective

The project successfully demonstrates an end-to-end fraud-scoring workflow designed around a fixed review-capacity constraint.

On the held-out test set, the final XGBoost model achieved:

- **PR-AUC: 0.9997**
- **Recall @ 1% FPR: 100.00%**
- **13,110 / 13,110 fraud transactions detected**
- **1,369 legitimate transactions flagged**

### What went well

- The evaluation metric was tied directly to the fraud-review business constraint.
- A simple rule-based baseline was established before ML.
- Feature engineering encoded interpretable fraud-operations logic.
- Training and serving reused the same transformation code.
- FastAPI deployment was included rather than stopping at model training.
- PSI monitoring and a retraining trigger were implemented.
- Limitations were explicitly documented.

### What to improve next

1. Replace the simulated dataset with real or appropriately anonymized transaction data.
2. Add transaction timestamps and perform temporal validation.
3. Add transaction amount and calculate actual financial impact.
4. Validate the model across multiple time periods.
5. Build a formal automated test suite.
6. Add fairness and compliance review where applicable.
7. Containerize and deploy through CI/CD.
8. Monitor real fraud-label feedback and concept drift.
9. Recalibrate the operating threshold as fraud prevalence and review capacity change.

### Final delivery

The project demonstrates the complete ML lifecycle:

**Business Requirement → Data Understanding → EDA → Feature Engineering → Model Development → Business-Constrained Evaluation → API Deployment → Testing → Monitoring → Maintenance Planning → Documentation**

---

## Project File Map

```text
day-02-credit-card-fraud-detection/
│
├── data/
│   └── card_transdata.csv
│
├── data_loader.py
├── eda.py
├── clean_and_engineer.py
├── split.py
├── train_model.py
├── app.py
├── monitor.py
├── README.md
├── PROJECT_DOCUMENTATION.md
└── SDLC_DOCUMENTATION.md
```

---

## Key Metrics at a Glance

| Metric | Result |
|---|---:|
| Dataset size | 1,000,000 transactions |
| Raw behavioral features | 7 |
| Final model features | 10 |
| Fraud rate | 8.74% |
| Final model | XGBoost |
| Test PR-AUC | 0.9997 |
| Recall @ 1% FPR | 100.00% |
| Selected threshold | 0.0018 |
| Test fraud detected | 13,110 / 13,110 |
| Legitimate transactions flagged | 1,369 |
| Rule-based recall | 70.5% |
| Rule-based precision | 66.6% |

---

## Critical Production Disclaimer

This project demonstrates **methodology**, not production fraud performance.

The dataset is simulated, contains an unusually high fraud prevalence, and has strong feature separation. Therefore:

- The reported 100% recall at 1% FPR is dataset-specific.
- The model should not be marketed as a perfect fraud detector.
- The illustrative $6.555M loss-prevention calculation is not actual measured business impact.
- Production deployment requires real transaction data, temporal validation, real fraud outcomes, transaction amounts, ongoing monitoring, and operational validation with fraud-review teams.
