"""
Complete preprocessing pipeline:
1. Outlier removal
2. Missing value imputation
3. Financial feature engineering
4. Categorical encoding
5. Train/test split (stratified)
6. SMOTE oversampling
7. Feature scaling
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE
from src.utils.logger        import get_logger

log = get_logger("data.preprocessor")

TARGET = "loan_status"

NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = [
    "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]


# Feature Engineering─────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 6 domain-driven financial risk indicators.

    debt_to_income_ratio
        Monthly loan payment proxy ÷ monthly income.
        Core metric used by every major lender — values >0.43
        are the standard "risky" threshold (US CFPB guideline).

    loan_to_income_ratio
        Total loan amount ÷ annual income.
        Captures how large the loan burden is relative to
        the applicant's earning capacity.

    income_per_cred_year
        Annual income ÷ years of credit history.
        Combines earning power with financial experience —
        a young high-earner with no credit history is riskier
        than the raw income figure suggests.

    high_risk_grade_flag
        Binary: 1 if loan grade is E, F, or G.
        Sub-prime grades default at 4–6× the rate of A/B grades.
        Captures lender-assessed risk in a single binary signal.

    int_rate_x_loan_amnt
        Interest rate × loan amount (interaction term).
        Total interest burden — high rate on a large loan is
        an extreme compounding risk factor.

    employment_stability_score
        Sigmoid-scaled employment length.
        Converts years (0–60) into a 0–1 score with diminishing
        returns — 10 yrs vs 1 yr matters more than 30 vs 20 yrs.
    """
    out = df.copy()

    income   = out["person_income"].replace(0, 1)
    cred_yrs = out["cb_person_cred_hist_length"].replace(0, 1)
    emp_len  = out["person_emp_length"].fillna(0)

    out["debt_to_income_ratio"]      = (out["loan_amnt"] / 12 / (income / 12)).round(4)
    out["loan_to_income_ratio"]      = (out["loan_amnt"] / income).round(4)
    out["income_per_cred_year"]      = (income / cred_yrs).round(2)
    out["high_risk_grade_flag"]      = out["loan_grade"].isin(["E","F","G"]).astype(int)
    out["int_rate_x_loan_amnt"]      = (out["loan_int_rate"].fillna(out["loan_int_rate"].median())
                                        * out["loan_amnt"] / 1000).round(4)
    out["employment_stability_score"]= (1 / (1 + np.exp(-0.3 * (emp_len - 5)))).round(4)

    log.info("Feature engineering complete")
    return out


class Preprocessor:
    """
    End-to-end preprocessing for the credit scoring dataset.

    Usage
    -----
    pp     = Preprocessor(cfg)
    splits = pp.fit_transform(df)
    # splits keys: X_train, X_test, y_train, y_test,
    #              X_train_sc, X_test_sc, feature_names
    """

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.scaler      = StandardScaler()
        self.feature_names_: list[str] = []

    # Step 1: Outlier removal
    @staticmethod
    def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["person_age"]        <= 100]
        df = df[df["person_income"]     <= 1_500_000]
        df = df[df["person_emp_length"] <= 60]
        log.info(f"Outliers removed: {before - len(df)} rows  "
                 f"({(before-len(df))/before*100:.1f}%)")
        return df.reset_index(drop=True)

    # Step 2: Missing value imputation
    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        miss = df.isnull().sum()
        if miss.any():
            cols = miss[miss > 0]
            for col in cols.index:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    log.info(f"Imputed '{col}' with median={med:.2f}")
                else:
                    mod = df[col].mode()[0]
                    df[col] = df[col].fillna(mod)
                    log.info(f"Imputed '{col}' with mode='{mod}'")
        else:
            log.info("No missing values after outlier removal")
        return df

    # Step 3: Encode categoricals
    @staticmethod
    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS,
                            drop_first=False, dtype=int)
        log.info(f"After one-hot encoding - shape: {df.shape}")
        return df

    # Full pipeline
    def fit_transform(self, df: pd.DataFrame) -> dict:
        log.info("Starting preprocessing pipeline")

        df = self._remove_outliers(df)
        df = self._impute(df)
        df = engineer_features(df)
        df = self._encode(df)

        feature_names = [c for c in df.columns if c != TARGET]
        self.feature_names_ = feature_names

        X = df[feature_names]
        y = df[TARGET]

        # ── Stratified split ──────────────────────────────────
        seed = self.cfg["data"]["random_seed"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg["data"]["test_size"],
            random_state=seed, stratify=y,
        )
        log.info(f"Split → train: {len(X_train):,}  test: {len(X_test):,}")
        log.info(f"Train balance before SMOTE: "
                 f"{dict(pd.Series(y_train).value_counts())}")

        # ── SMOTE ─────────────────────────────────────────────
        if self.cfg["data"].get("use_smote", True):
            k = self.cfg["data"].get("smote_k_neighbors", 5)
            sm = SMOTE(random_state=seed, k_neighbors=k)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            log.info(f"SMOTE applied  → train: {len(X_res):,}  "
                     f"balance: {dict(pd.Series(y_res).value_counts())}")
            X_train = pd.DataFrame(X_res, columns=feature_names)
            y_train = pd.Series(y_res, name=TARGET)

        # ── Scale ─────────────────────────────────────────────
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)
        log.info("Features scaled with StandardScaler  ✓")

        log.info(f"Final feature count: {len(feature_names)}")

        return {
            "X_train":       X_train,
            "X_test":        X_test,
            "y_train":       y_train,
            "y_test":        y_test,
            "X_train_sc":    X_train_sc,
            "X_test_sc":     X_test_sc,
            "feature_names": feature_names,
        }
