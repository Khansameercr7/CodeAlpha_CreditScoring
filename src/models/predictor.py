"""
Real-time single-applicant credit risk prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.preprocessor import engineer_features, CATEGORICAL_COLS
from src.utils.logger       import get_logger

log = get_logger("models.predictor")


class CreditPredictor:
    """
    Wraps any trained model for live inference.

    Replicates the exact preprocessing applied during training:
    feature engineering → one-hot encoding → column alignment → scaling.
    """

    def __init__(self, model, scaler: StandardScaler,
                 feature_names: list[str], use_scaling: bool = False):
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self.use_scaling   = use_scaling

    def _prepare(self, raw: dict) -> pd.DataFrame:
        row = pd.DataFrame([raw])
        # Fill missing fields that engineering needs
        for col in ["loan_int_rate","person_emp_length"]:
            if col not in row.columns or row[col].isnull().all():
                row[col] = 0
        row = engineer_features(row)
        # One-hot encode categoricals present in input
        cats = [c for c in CATEGORICAL_COLS if c in row.columns]
        if cats:
            row = pd.get_dummies(row, columns=cats, drop_first=False, dtype=int)
        # Align to training columns
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0
        return row[self.feature_names]

    def predict(self, raw_input: dict, threshold: float = 0.5) -> dict:
        """
        Predict credit risk for one applicant.

        Parameters
        ----------
        raw_input : dict of loan application fields
        threshold : probability threshold for default classification

        Returns
        -------
        dict with label, probability, risk_band, recommendation,
             risk_score (0–100) and key_risk_factors
        """
        X = self._prepare(raw_input)
        X_in = self.scaler.transform(X) if self.use_scaling else X.values

        prob_default = float(self.model.predict_proba(X_in)[0, 1])
        label        = int(prob_default >= threshold)
        label_str    = "Default Risk" if label == 1 else "Good Standing"
        risk_score   = round(prob_default * 100, 1)

        if prob_default < 0.20:   risk_band = "Very Low Risk"
        elif prob_default < 0.40: risk_band = "Low Risk"
        elif prob_default < 0.60: risk_band = "Medium Risk"
        elif prob_default < 0.80: risk_band = "High Risk"
        else:                     risk_band = "Very High Risk"

        recommendation = (
            "Reject - High default probability. Consider additional collateral."
            if label == 1
            else "Approve - Applicant appears creditworthy."
        )

        # Simple key risk factors from input
        risk_factors = []
        if raw_input.get("loan_grade","A") in ["E","F","G"]:
            risk_factors.append("Sub-prime loan grade (E/F/G)")
        if raw_input.get("cb_person_default_on_file","N") == "Y":
            risk_factors.append("Prior default on record")
        if raw_input.get("loan_percent_income", 0) > 0.3:
            risk_factors.append("Loan exceeds 30% of income")
        if raw_input.get("person_emp_length", 10) < 1:
            risk_factors.append("Less than 1 year employment")
        if not risk_factors:
            risk_factors.append("No major risk flags detected")

        result = {
            "label":            label_str,
            "probability":      round(prob_default, 4),
            "risk_score":       risk_score,
            "risk_band":        risk_band,
            "recommendation":   recommendation,
            "key_risk_factors": risk_factors,
        }
        log.info(f"Prediction → {label_str}  P(default)={prob_default:.4f}  {risk_band}")
        return result
