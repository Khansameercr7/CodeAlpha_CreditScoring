"""
SHAP model explainability - summary, bar, and waterfall plots.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.utils.logger import get_logger

log = get_logger("visualization.shap")


class ShapExplainer:
    """
    Generates SHAP explanations for each trained model.

    Why SHAP?─
    SHAP (SHapley Additive exPlanations) assigns each feature a
    contribution score for every individual prediction.

    • Summary plot  — shows global feature importance + direction
    • Bar plot      — mean |SHAP| — average impact magnitude
    • Waterfall     — single-applicant prediction breakdown
      (required by EU AI Act & US Fair Lending regulations)
    """

    def __init__(self, out_dir: str = "outputs/shap"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _get_explainer(self, model, X_bg):
        name = type(model).__name__
        if name in ("RandomForestClassifier","DecisionTreeClassifier",
                    "XGBClassifier","LGBMClassifier"):
            return shap.TreeExplainer(model)
        return shap.LinearExplainer(model, X_bg)

    def explain(self, model, model_name: str,
                X_test, feature_names: list, max_samples: int = 300) -> None:
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        X_s = X_test.iloc[:max_samples]
        slug = model_name.lower().replace(" ","_")

        log.info(f"  Computing SHAP for {model_name} …")
        try:
            explainer   = self._get_explainer(model, X_s)
            shap_values = explainer.shap_values(X_s)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # ── Summary beeswarm ─────────────────────────────
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(shap_values, X_s, feature_names=feature_names,
                              show=False, max_display=15)
            plt.title(f"SHAP Summary — {model_name}",
                      fontsize=13, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"shap_summary_{slug}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"    → {p}")

            # ── Bar (mean |SHAP|) ─────────────────────────────
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.summary_plot(shap_values, X_s, feature_names=feature_names,
                              plot_type="bar", show=False, max_display=15)
            plt.title(f"SHAP Feature Importance — {model_name}",
                      fontsize=13, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"shap_bar_{slug}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"    → {p}")

            # ── Waterfall (applicant #1) ──────────────────────
            # Handle 2D shap_values from tree classifiers (shape [n, classes])
            if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                sv0 = shap_values[0, :, 1]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
                sv0 = shap_values[0]
            else:
                sv0 = shap_values[0]
            base = (explainer.expected_value
                    if not isinstance(explainer.expected_value, (list,np.ndarray))
                    else float(explainer.expected_value[1]))
            expl = shap.Explanation(values=sv0, base_values=float(base),
                                    data=X_s.iloc[0].values,
                                    feature_names=feature_names)
            fig, _ = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(expl, show=False, max_display=15)
            plt.title(f"SHAP Waterfall (Applicant #1) — {model_name}",
                      fontsize=12, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(self.out_dir, f"shap_waterfall_{slug}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"    → {p}")

        except Exception as e:
            log.warning(f"SHAP failed for {model_name}: {e}")

    def explain_all(self, models: dict, splits: dict) -> None:
        log.info("Running SHAP explainability …")
        X_test    = splits["X_test"]
        X_test_sc = splits["X_test_sc"]
        feat      = splits["feature_names"]
        for name, model in models.items():
            X = X_test_sc if name == "Logistic Regression" else X_test
            self.explain(model, name, X, feat)
        log.info(f"SHAP plots saved to {self.out_dir}/")
