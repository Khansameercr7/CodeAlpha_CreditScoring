"""
Loads the real credit dataset and prints exploratory data analysis.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger("data.loader")


def load_and_explore(cfg: dict) -> pd.DataFrame:
    """
    Load the credit CSV and print exploratory data analysis.

    Dataset columns
    ───────────────
    person_age               Age of applicant
    person_income            Annual income (USD)
    person_home_ownership    RENT / OWN / MORTGAGE / OTHER
    person_emp_length        Employment length in years
    loan_intent              Purpose of loan
    loan_grade               Lender-assigned credit grade (A–G)
    loan_amnt                Loan amount requested
    loan_int_rate            Interest rate on loan (%)
    loan_status              TARGET — 0 = Good Standing, 1 = Default
    loan_percent_income      Loan amount as % of annual income
    cb_person_default_on_file  Prior default on record (Y/N)
    cb_person_cred_hist_length  Credit history length (years)
    """
    path = cfg["data"]["raw_path"]
    log.info(f"Loading dataset from '{path}' …")
    df = pd.read_csv(path)

    log.info(f"Loaded shape={df.shape}")
    log.info(f"Target balance: {dict(df['loan_status'].value_counts())} (0=Good, 1=Default)")

    print("\nExploratory Data Analysis")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    vc = df["loan_status"].value_counts()
    pct_default = vc[1] / len(df) * 100
    print(f"\nClass Balance:")
    print(f"Good Standing (0): {vc[0]:>6,} ({100-pct_default:.1f}%)")
    print(f"    Default       (1) : {vc[1]:>6,}  ({pct_default:.1f}%)")
    print(f"    → Imbalance ratio : {vc[0]/vc[1]:.1f}:1  (SMOTE will balance this)")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n  Missing Values:")
        for col, cnt in missing[missing > 0].items():
            print(f"    {col:<35} {cnt:>4} ({cnt/len(df)*100:.1f}%)")

    # Outliers summary
    print(f"\n  Outlier Check:")
    for col in ["person_age","person_income","person_emp_length"]:
        q99 = df[col].quantile(0.99)
        over = (df[col] > q99).sum()
        print(f"    {col:<30} 99th pct={q99:.0f}  |  extreme values={over}")

    # Categorical distributions
    print(f"\n  Categorical Distributions:")
    for col in ["person_home_ownership","loan_intent","loan_grade",
                "cb_person_default_on_file"]:
        vc2 = df[col].value_counts()
        print(f"    {col}:")
        for k, v in vc2.items():
            print(f"      {str(k):<20} {v:>5,}  ({v/len(df)*100:.1f}%)")

    # Numeric stats
    print(f"\n  Numeric Statistics:")
    print(df.describe().round(2).to_string())
    print()

    return df
