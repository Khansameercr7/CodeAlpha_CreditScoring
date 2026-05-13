"""
Industry-standard financial risk metrics and cross-validation.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
)
from sklearn.model_selection import cross_val_score
from src.utils.logger import get_logger

log = get_logger("evaluation.evaluator")

NEEDS_SCALING = {"Logistic Regression"}


class ModelEvaluator:
    """
    Evaluates all models using financial credit risk metrics.

    Metrics explained:─
    Accuracy   : % of all predictions correct.
                 Misleading on imbalanced data — a model predicting
                 all "Good" gets 78% accuracy but is useless.

    Precision  : Of applicants flagged as default, what % actually default.
                 Low precision = too many false alarms (good applicants rejected).

    Recall     : Of actual defaults, what % did the model catch.
                 Low recall = missed defaults (financial loss to lender).

    F1-Score   : Harmonic mean of Precision and Recall.
                 Best single metric for imbalanced classification.

    ROC-AUC    : Area under the ROC curve. Measures rank ordering ability.
                 0.5 = random, 1.0 = perfect. Industry standard ≥ 0.75.

    PR-AUC     : Area under Precision-Recall curve.
                 More informative than ROC-AUC on heavily imbalanced data.
    """

    def __init__(self, cv_folds: int = 5, threshold: float = 0.50):
        self.cv_folds   = cv_folds
        self.threshold  = threshold
        self.results_   : list[dict] = []
        self.compare_df : pd.DataFrame | None = None

    def _score(self, name: str, model, X, y) -> dict:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return {
            "Model":    name,
            "Accuracy": round(accuracy_score(y, y_pred),                    4),
            "Precision":round(precision_score(y, y_pred, zero_division=0),  4),
            "Recall":   round(recall_score(y, y_pred,    zero_division=0),  4),
            "F1-Score": round(f1_score(y, y_pred,        zero_division=0),  4),
            "ROC-AUC":  round(roc_auc_score(y, y_prob),                     4),
            "PR-AUC":   round(average_precision_score(y, y_prob),           4),
            "_y_pred":  y_pred,
            "_y_prob":  y_prob,
        }

    def evaluate(self, models: dict, splits: dict) -> pd.DataFrame:
        X_test, X_test_sc = splits["X_test"], splits["X_test_sc"]
        y_test = splits["y_test"]

        self.results_ = []
        for name, model in models.items():
            X = X_test_sc if name in NEEDS_SCALING else X_test
            r = self._score(name, model, X, y_test)
            self.results_.append(r)
            log.info(
                f"{name:<22}  Acc={r['Accuracy']:.4f}  "
                f"F1={r['F1-Score']:.4f}  AUC={r['ROC-AUC']:.4f}  "
                f"PR-AUC={r['PR-AUC']:.4f}"
            )

        self.compare_df = pd.DataFrame(
            [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in self.results_]
        )
        best = self.compare_df.loc[self.compare_df["ROC-AUC"].idxmax(), "Model"]
        log.info(f"★  Best model by ROC-AUC → {best}")
        return self.compare_df

    def cross_validate(self, models: dict, splits: dict) -> None:
        X_train, X_train_sc = splits["X_train"], splits["X_train_sc"]
        y_train = splits["y_train"]
        log.info(f"\n{self.cv_folds}-Fold Cross-Validation:")
        for name, model in models.items():
            X = X_train_sc if name in NEEDS_SCALING else X_train
            for metric in ["f1", "roc_auc"]:
                s = cross_val_score(model, X, y_train,
                                    cv=self.cv_folds, scoring=metric, n_jobs=-1)
                log.info(f"  {name:<22} {metric:<8}  "
                         f"mean={s.mean():.4f}  std={s.std():.4f}")

    def print_best_report(self, splits: dict) -> None:
        idx    = self.compare_df["ROC-AUC"].idxmax()
        best   = self.results_[idx]
        y_test = splits["y_test"]
        print(f"\nDetailed Report — {best['Model']}:")
        print(classification_report(y_test, best["_y_pred"],
              target_names=["Good Standing","Default"]))

    def print_false_positive_analysis(self) -> None:
        print("""
  ── False Positive vs False Negative — Business Impact ──────

  FALSE POSITIVE (Predict Default → Actually Good Standing):
    • The bank rejects a creditworthy applicant.
    • Revenue loss: missed loan interest income.
    • Reputational risk: applicant may complain or go to competitor.
    • Regulatory risk: fair-lending law violations if systemic.
    → Reduce by LOWERING the decision threshold.

  FALSE NEGATIVE (Predict Good → Actually Defaults):
    • The bank approves a loan that goes bad.
    • Direct financial loss: principal + lost interest.
    • For a £10,000 loan at 15% over 3 years, a default costs ~£12,000.
    → Reduce by RAISING the decision threshold.

  Industry Guidance:
    • Most lenders prioritise Recall (catching defaults) over Precision.
    • Typical operating threshold: 0.3–0.4 (not 0.5) for credit risk.
    • Precision-Recall curves (not ROC) drive operational decisions.
        """)

    def save_metrics(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.compare_df.to_csv(path, index=False)
        log.info(f"Metrics CSV saved → {path}")
