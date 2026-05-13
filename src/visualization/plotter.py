"""
Project visualizations
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.tree    import plot_tree
from src.utils.logger import get_logger

log = get_logger("visualization.plotter")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2ecc71","#e74c3c"]          # Good / Default
MODEL_COLORS = ["#3498db","#e67e22","#9b59b6","#e74c3c"]


class Plotter:
    def __init__(self, out_dir: str = "outputs/figures"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _save(self, fig, name: str) -> None:
        p = os.path.join(self.out_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved → {p}")

    # Fig 1: Class distribution────
    def plot_class_distribution(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Target Variable — Loan Status Distribution",
                     fontsize=14, fontweight="bold")

        vc = df["loan_status"].value_counts()
        labels = ["Good Standing (0)", "Default (1)"]

        axes[0].bar(labels, vc.values, color=COLORS, edgecolor="white", width=0.5)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)",
                         ha="center", fontweight="bold")
        axes[0].set_title("Absolute Count", fontweight="bold")
        axes[0].set_ylabel("Count")

        axes[1].pie(vc.values, labels=labels, colors=COLORS,
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor":"white","linewidth":2})
        axes[1].set_title("Proportion", fontweight="bold")

        plt.tight_layout()
        self._save(fig, "fig01_class_distribution.png")

    # ── Fig 2 : Numeric feature distributions ────────────────
    def plot_numeric_distributions(self, df: pd.DataFrame) -> None:
        numeric_cols = [
            "person_age","person_income","person_emp_length",
            "loan_amnt","loan_int_rate","loan_percent_income",
            "cb_person_cred_hist_length",
        ]
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle("Numeric Feature Distributions by Credit Status",
                     fontsize=14, fontweight="bold")

        for ax, col in zip(axes.flatten(), numeric_cols):
            for status, label, color in [(0,"Good Standing","#2ecc71"),
                                          (1,"Default","#e74c3c")]:
                data = df[df["loan_status"]==status][col].dropna()
                ax.hist(data, bins=30, alpha=0.6, label=label,
                        color=color, edgecolor="white", density=True)
            ax.set_title(col.replace("_"," ").title(), fontweight="bold", fontsize=10)
            ax.legend(fontsize=8)

        # hide unused subplot
        axes.flatten()[-1].set_visible(False)
        axes.flatten()[-2].set_visible(False)
        plt.tight_layout()
        self._save(fig, "fig02_numeric_distributions.png")

    # ── Fig 3 : Categorical features ─────────────────────────
    def plot_categorical_features(self, df: pd.DataFrame) -> None:
        cat_cols = ["person_home_ownership","loan_intent",
                    "loan_grade","cb_person_default_on_file"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Categorical Features — Default Rate Analysis",
                     fontsize=14, fontweight="bold")

        for ax, col in zip(axes.flatten(), cat_cols):
            rate = df.groupby(col)["loan_status"].mean().sort_values(ascending=False)
            bars = ax.bar(rate.index, rate.values * 100,
                          color=plt.cm.RdYlGn_r(rate.values), edgecolor="white")
            ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, fontweight="bold")
            ax.set_title(f"{col.replace('_',' ').title()} — Default Rate",
                         fontweight="bold")
            ax.set_ylabel("Default Rate (%)")
            ax.set_ylim(0, rate.max()*100 + 15)
            ax.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        self._save(fig, "fig03_categorical_default_rates.png")

    # ── Fig 4 : Correlation heatmap ───────────────────────────
    def plot_correlation(self, df: pd.DataFrame) -> None:
        num_cols = [
            "person_age","person_income","person_emp_length",
            "loan_amnt","loan_int_rate","loan_percent_income",
            "cb_person_cred_hist_length","loan_status",
        ]
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, linewidths=0.5, ax=ax,
                    annot_kws={"size": 10})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "fig04_correlation_heatmap.png")

    # ── Fig 5 : Engineered features ───────────────────────────
    def plot_engineered_features(self, df_proc: pd.DataFrame) -> None:
        eng_cols = [c for c in [
            "debt_to_income_ratio","loan_to_income_ratio",
            "income_per_cred_year","employment_stability_score",
        ] if c in df_proc.columns]

        if not eng_cols:
            return

        fig, axes = plt.subplots(1, len(eng_cols), figsize=(5*len(eng_cols), 5))
        if len(eng_cols) == 1:
            axes = [axes]
        fig.suptitle("Engineered Financial Risk Features",
                     fontsize=14, fontweight="bold")

        for ax, col in zip(axes, eng_cols):
            for status, label, color in [(0,"Good","#2ecc71"),(1,"Default","#e74c3c")]:
                if "loan_status" in df_proc.columns:
                    data = df_proc[df_proc["loan_status"]==status][col].dropna()
                else:
                    data = df_proc[col].dropna()
                ax.hist(data, bins=30, alpha=0.6, label=label,
                        color=color, edgecolor="white", density=True)
            ax.set_title(col.replace("_"," ").title(), fontweight="bold", fontsize=10)
            ax.legend(fontsize=9)

        plt.tight_layout()
        self._save(fig, "fig05_engineered_features.png")

    # ── Fig 6 : ROC curves ────────────────────────────────────
    def plot_roc_curves(self, results: list, y_test) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        for r, col in zip(results, MODEL_COLORS):
            fpr, tpr, _ = roc_curve(y_test, r["_y_prob"])
            ax.plot(fpr, tpr, lw=2.5, color=col,
                    label=f"{r['Model']}  (AUC={r['ROC-AUC']:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1.5,label="Random Classifier (AUC=0.500)")
        ax.fill_between([0,1],[0,1],alpha=0.04,color="grey")
        ax.set_xlabel("False Positive Rate (1 - Specificity)",fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)",fontsize=12)
        ax.set_title("ROC Curves — All Models",fontsize=14,fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.02])
        plt.tight_layout()
        self._save(fig, "fig06_roc_curves.png")

    # ── Fig 7 : Precision-Recall curves ──────────────────────
    def plot_pr_curves(self, results: list, y_test) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        baseline = y_test.mean()
        for r, col in zip(results, MODEL_COLORS):
            prec, rec, _ = precision_recall_curve(y_test, r["_y_prob"])
            ax.plot(rec, prec, lw=2.5, color=col,
                    label=f"{r['Model']}  (PR-AUC={r['PR-AUC']:.3f})")
        ax.axhline(baseline, color="grey", linestyle="--", lw=1.5,
                   label=f"No-skill baseline ({baseline:.3f})")
        ax.set_xlabel("Recall",fontsize=12)
        ax.set_ylabel("Precision",fontsize=12)
        ax.set_title("Precision-Recall Curves — All Models",fontsize=14,fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, "fig07_pr_curves.png")

    # ── Fig 8 : Confusion matrices ────────────────────────────
    def plot_confusion_matrices(self, results: list, y_test) -> None:
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        fig.suptitle("Confusion Matrices",fontsize=14,fontweight="bold")
        cmaps = ["Blues","Oranges","Purples","Reds"]
        for ax, r, cmap in zip(axes, results, cmaps):
            cm = confusion_matrix(y_test, r["_y_pred"])
            ConfusionMatrixDisplay(cm, display_labels=["Good","Default"]).plot(
                ax=ax, colorbar=False, cmap=cmap)
            ax.set_title(r["Model"], fontweight="bold", fontsize=10)
        plt.tight_layout()
        self._save(fig, "fig08_confusion_matrices.png")

    # ── Fig 9 : Model comparison bar chart ───────────────────
    def plot_model_comparison(self, compare_df: pd.DataFrame) -> None:
        metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]
        x = np.arange(len(metrics))
        width = 0.20
        fig, ax = plt.subplots(figsize=(15, 6))
        for i, (_, row) in enumerate(compare_df.iterrows()):
            vals = [row[m] for m in metrics]
            bars = ax.bar(x + i*width, vals, width,
                          label=row["Model"], color=MODEL_COLORS[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0.55, 1.05)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, "fig09_model_comparison.png")

    # ── Fig 10 : Feature importance (RF + XGBoost) ───────────
    def plot_feature_importance(self, models: dict, feature_names: list) -> None:
        target_models = {k: v for k, v in models.items()
                         if k in ("Random Forest","XGBoost")}
        n = len(target_models)
        fig, axes = plt.subplots(1, n, figsize=(12*n//2+4, 8))
        if n == 1: axes = [axes]
        fig.suptitle("Feature Importance",fontsize=14,fontweight="bold")
        colors = {"Random Forest":"#9b59b6","XGBoost":"#e74c3c"}

        for ax, (name, model) in zip(axes, target_models.items()):
            imp = model.feature_importances_
            fi = pd.DataFrame({"Feature":feature_names,"Importance":imp})
            fi = fi.sort_values("Importance",ascending=True).tail(20)
            bars = ax.barh(fi["Feature"], fi["Importance"],
                           color=colors.get(name,"#3498db"),
                           edgecolor="white", alpha=0.85)
            ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
            ax.set_xlabel("Importance Score", fontsize=11)
            ax.set_title(f"{name} — Top 20 Features", fontweight="bold")
            ax.set_xlim(0, fi["Importance"].max()*1.2)

        plt.tight_layout()
        self._save(fig, "fig10_feature_importance.png")

    # ── Fig 11 : Decision tree (top 3 levels) ────────────────
    def plot_decision_tree(self, dt_model, feature_names: list) -> None:
        fig, ax = plt.subplots(figsize=(22, 9))
        plot_tree(dt_model, feature_names=feature_names,
                  class_names=["Good","Default"],
                  max_depth=3, filled=True, rounded=True,
                  fontsize=8, ax=ax)
        ax.set_title("Decision Tree — First 3 Levels",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "fig11_decision_tree_structure.png")

    # ── Fig 12 : Threshold analysis ───────────────────────────
    def plot_threshold_analysis(self, results: list, y_test) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        thresholds = np.linspace(0.1, 0.9, 81)

        best = max(results, key=lambda r: r["ROC-AUC"])
        probs = best["_y_prob"]

        from sklearn.metrics import precision_score, recall_score, f1_score
        precs, recs, f1s = [], [], []
        for t in thresholds:
            pred = (probs >= t).astype(int)
            precs.append(precision_score(y_test, pred, zero_division=0))
            recs.append(recall_score(y_test, pred, zero_division=0))
            f1s.append(f1_score(y_test, pred, zero_division=0))

        ax.plot(thresholds, precs, label="Precision", color="#3498db", lw=2)
        ax.plot(thresholds, recs,  label="Recall",    color="#e74c3c", lw=2)
        ax.plot(thresholds, f1s,   label="F1-Score",  color="#2ecc71", lw=2)

        best_t = thresholds[np.argmax(f1s)]
        ax.axvline(best_t, color="grey", linestyle="--", lw=1.5,
                   label=f"Optimal threshold = {best_t:.2f}")

        ax.set_xlabel("Decision Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Threshold Optimisation — {best['Model']}",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        self._save(fig, "fig12_threshold_analysis.png")

    # ── Run all ───────────────────────────────────────────────
    def plot_all(self, df, df_eng, results, compare_df, splits, models) -> None:
        log.info("Generating all figures …")
        self.plot_class_distribution(df)
        self.plot_numeric_distributions(df)
        self.plot_categorical_features(df)
        self.plot_correlation(df)
        self.plot_engineered_features(df_eng)
        self.plot_roc_curves(results, splits["y_test"])
        self.plot_pr_curves(results, splits["y_test"])
        self.plot_confusion_matrices(results, splits["y_test"])
        self.plot_model_comparison(compare_df)
        self.plot_feature_importance(models, splits["feature_names"])
        self.plot_decision_tree(models["Decision Tree"], splits["feature_names"])
        self.plot_threshold_analysis(results, splits["y_test"])
        log.info(f"All 12 figures saved to {self.out_dir}/")
