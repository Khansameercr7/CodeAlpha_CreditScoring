# Credit Scoring & Loan Default Prediction

I built a credit risk prediction system that classifies loan applicants as likely to default or not. It uses scikit-learn, XGBoost, and Streamlit with SHAP explainability to show why the model makes its decisions.

## What This Project Does

Basically, I trained machine learning models on ~32K loan records to predict which applicants will default. The tricky part was getting the models to actually explain their predictions—not just give yes/no answers. That's where SHAP came in. I also built a simple web dashboard so someone could paste in applicant details and see predictions in real-time.

**What I Actually Built:**
- 4 different models (Logistic Regression, Decision Tree, Random Forest, XGBoost) and compared them
- Custom financial features (debt-to-income ratio, payment history, credit utilization)
- A Streamlit web app where you can enter applicant info and get a live prediction
- SHAP visualizations that show which factors pushed the prediction up or down
- Proper model evaluation with ROC curves, precision-recall, and cross-validation
- Handled imbalanced data with SMOTE since defaults are rare

## What I Learned Building This

1. **Binary Classification** - Predicting defaults is a classification problem; I learned how to frame it, evaluate it properly, and handle class imbalance
2. **Feature Engineering** - Not all features matter equally. I crafted financial metrics that actually make sense to loan officers
3. **Model Comparison** - Tried 4 different algorithms and tuned them with Optuna to find the best performer
4. **Model Explainability** - SHAP really changed how I think about ML. Now I can show exactly which features matter for each prediction
5. **Production Code** - Learned to organize code into modules (data/, models/, evaluation/), use config files instead of hardcoding, and write tests
6. **Imbalanced Data** - Defaults are rare, so SMOTE was essential to avoid training a model that just predicts "no default" all the time
7. **Data Cleaning** - Outliers, missing values, encoding—it all takes time but quality preprocessing makes models better
8. **Building UIs** - Streamlit made it surprisingly easy to build a web app so non-technical people could use the model

## Project Structure

```
credit_scoring/
│
├── main.py                          ← Run full pipeline
├── app.py                           ← Streamlit dashboard
├── requirements.txt
├── README.md
│
├── config/
│   └── config.yaml                     ← All settings (no magic numbers)
│
├── data/
│   ├── credit_data.csv                 ← Raw dataset (32,581 rows)
│   └── processed_data.csv              ← Auto-generated after main.py
│
├── src/
│   ├── data/
│   │   ├── loader.py                   ← Load + full EDA report
│   │   └── preprocessor.py            ← Outliers→Impute→Engineer→Encode→SMOTE→Scale
│   ├── models/
│   │   ├── trainer.py                  ← LR, DT, RF, XGBoost
│   │   ├── tuner.py                    ← Optuna Bayesian tuning
│   │   └── predictor.py               ← Live single-applicant inference
│   ├── evaluation/
│   │   └── evaluator.py               ← 6 metrics + CV + FP/FN analysis
│   ├── visualization/
│   │   ├── plotter.py                  ← 12 standard figures
│   │   └── shap_explainer.py          ← SHAP summary, bar & waterfall
│   └── utils/
│       ├── logger.py                   ← Colour-coded logger
│       └── model_io.py                ← joblib save/load
│
├── tests/
│   └── test_pipeline.py               ← 25+ unit tests
│
└── outputs/
    ├── figures/                        ← 12 PNG charts (auto-generated)
    ├── shap/                           ← SHAP plots (auto-generated)
    ├── models/                         ← 4 .pkl models + scaler + features
    └── reports/
        └── model_metrics.csv          ← Comparison table
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full ML pipeline
python main.py

# 3. Launch web dashboard
streamlit run app.py

# 4. Run unit tests
python -m pytest tests/ -v
```

---

## Dataset

| Field | Value |
|---|---|
| Source | UCI Default of Credit Card / Loan Dataset |
| Rows | 32,581 applicants |
| Features | 12 raw → 32 after encoding |
| Target | `loan_status` (0=Good Standing, 1=Default) |
| Class Balance | 78.2% Good / 21.8% Default → corrected with SMOTE |
| Missing Values | `loan_int_rate` (9.6%), `person_emp_length` (2.7%) |

---

## Feature Engineering

| Feature | Formula | Why |
|---|---|---|
| `debt_to_income_ratio` | loan/12 ÷ income/12 | Core affordability metric (CFPB threshold: 0.43) |
| `loan_to_income_ratio` | loan_amnt ÷ income | Borrowing burden vs earnings |
| `income_per_cred_year` | income ÷ cred_years | Financial maturity proxy |
| `high_risk_grade_flag` | 1 if grade ∈ {E,F,G} | Sub-prime default at 4–6× |
| `int_rate_x_loan_amnt` | rate × amount / 1000 | Total interest burden |
| `employment_stability_score` | sigmoid(emp_length) | Non-linear job stability |

---

## Models

| Model | How it works |
|---|---|
| **Logistic Regression** | Sigmoid boundary on linear feature combination |
| **Decision Tree** | Recursive Gini splits → human-readable rules |
| **Random Forest** | 300 trees, bagging + feature randomness, majority vote |
| **XGBoost** | Sequential gradient boosting, corrects residual errors |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.879 | 0.782 | 0.606 | 0.683 | 0.880 | 0.754 |
| Decision Tree | 0.919 | 0.900 | 0.703 | 0.789 | 0.908 | 0.844 |
| Random Forest | 0.925 | 0.896 | 0.739 | 0.810 | 0.920 | 0.870 |
| **XGBoost** | **0.888** | 0.711 | **0.810** | 0.758 | **0.937** | **0.887** |

**Best model: XGBoost** — ROC-AUC = 0.937 (industry target ≥ 0.75)

---

## Streamlit Dashboard Pages

| Page | Description |
|---|---|
| Overview | System summary, pipeline steps, feature explanations |
| Risk Predictor | Live applicant assessment with probability gauge |
| Model Performance | Comparison table, metric deep-dive, FP/FN analysis |
| Charts & Figures | All 12 visualisation figures |
| SHAP Explainability | Summary, bar, and waterfall SHAP plots |
| Dataset Explorer | Raw data, statistics, interactive distributions |

---

## Tests

```bash
python -m pytest tests/ -v
# Expected: 25+ tests passed
```

Test coverage:
- Data loading & outlier removal
- Preprocessing, SMOTE, encoding, scaling
- Feature engineering correctness
- All 4 model outputs
- Evaluation metrics range
- Model save/load (joblib round-trip)
- Predictor: low-risk vs high-risk applicants

---

## Configuration

All hyperparameters and paths live in `config/config.yaml`.
No magic numbers in source code.

Key settings:
```yaml
data.use_smote: true          # enable/disable SMOTE
tuning.enabled: true          # enable/disable Optuna tuning
tuning.n_trials: 40           # Optuna trials
evaluation.threshold: 0.50    # decision boundary
```

---

## Tech Stack

`Python 3.9+` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `SHAP`
`pandas` · `numpy` · `matplotlib` · `seaborn` · `Streamlit` · `Optuna` · `joblib`

---

## Future Enhancements

- Real-time credit scoring REST API (FastAPI)
- Deep Learning risk analysis (TabNet)
- Fraud detection integration
- Cloud deployment (AWS SageMaker / GCP Vertex AI)
- Automated model retraining pipeline
- Fairness audit (fairlearn)
- Probability calibration (Platt scaling)
