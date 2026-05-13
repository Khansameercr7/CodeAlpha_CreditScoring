"""
Trains Logistic Regression, Decision Tree, Random Forest,
and XGBoost classifiers.
"""
from __future__ import annotations
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier
from src.utils.logger      import get_logger

log = get_logger("models.trainer")

# Models that require scaled input
NEEDS_SCALING = {"Logistic Regression"}


class ModelTrainer:
    """
    Builds and trains all classifiers from config values.

    How each model works
    ────────────────────
    Logistic Regression
      Fits a logistic (sigmoid) curve to the linear combination
      of features. Outputs P(default). Fast, interpretable,
      requires feature scaling. Good baseline.

    Decision Tree
      Recursively partitions the feature space by the split that
      maximises information gain (Gini impurity). Produces human-
      readable if-else rules. Prone to overfitting without depth
      constraints.

    Random Forest
      Builds hundreds of Decision Trees on random bootstrap samples
      with random feature subsets (bagging + feature randomness).
      Majority vote across trees dramatically reduces variance.
      Robust, handles non-linearity, provides feature importance.

    XGBoost
      Sequential boosting: each new tree corrects the residual
      errors of all previous trees using gradient descent on the
      loss function. Uses second-order gradients for precise
      optimisation. State-of-the-art on tabular financial data.
      Handles class imbalance via scale_pos_weight parameter.
    """

    def __init__(self, cfg: dict, random_seed: int = 42):
        self.cfg         = cfg
        self.random_seed = random_seed
        self.models: dict = {}

    def _lr(self) -> LogisticRegression:
        c = self.cfg.get("logistic_regression", {})
        return LogisticRegression(
            max_iter     = c.get("max_iter", 1000),
            C            = c.get("C", 1.0),
            solver       = c.get("solver", "lbfgs"),
            class_weight = c.get("class_weight", "balanced"),
            random_state = self.random_seed,
        )

    def _dt(self) -> DecisionTreeClassifier:
        c = self.cfg.get("decision_tree", {})
        return DecisionTreeClassifier(
            max_depth         = c.get("max_depth", 8),
            min_samples_split = c.get("min_samples_split", 20),
            min_samples_leaf  = c.get("min_samples_leaf", 10),
            criterion         = c.get("criterion", "gini"),
            class_weight      = c.get("class_weight", "balanced"),
            random_state      = self.random_seed,
        )

    def _rf(self, override: dict | None = None) -> RandomForestClassifier:
        c = {**self.cfg.get("random_forest", {}), **(override or {})}
        return RandomForestClassifier(
            n_estimators      = c.get("n_estimators", 300),
            max_depth         = c.get("max_depth", 10),
            min_samples_split = c.get("min_samples_split", 10),
            min_samples_leaf  = c.get("min_samples_leaf", 5),
            class_weight      = c.get("class_weight", "balanced"),
            n_jobs            = c.get("n_jobs", -1),
            random_state      = self.random_seed,
        )

    def _xgb(self, override: dict | None = None) -> XGBClassifier:
        c = {**self.cfg.get("xgboost", {}), **(override or {})}
        return XGBClassifier(
            n_estimators      = c.get("n_estimators", 300),
            max_depth         = c.get("max_depth", 6),
            learning_rate     = c.get("learning_rate", 0.05),
            subsample         = c.get("subsample", 0.8),
            colsample_bytree  = c.get("colsample_bytree", 0.8),
            scale_pos_weight  = c.get("scale_pos_weight", 3.58),
            eval_metric       = "logloss",
            random_state      = self.random_seed,
            verbosity         = 0,
        )

    def train(self, splits: dict, tuned_params: dict | None = None) -> dict:
        """
        Train all 4 models.  If tuned_params is provided,
        they override config values for XGBoost.

        Returns
        -------
        dict[str, fitted estimator]
        """
        X_train    = splits["X_train"]
        y_train    = splits["y_train"]
        X_train_sc = splits["X_train_sc"]

        log.info("Training all models …")

        lr = self._lr()
        lr.fit(X_train_sc, y_train)
        log.info("  [1/4] Logistic Regression    ✓")

        dt = self._dt()
        dt.fit(X_train, y_train)
        log.info("  [2/4] Decision Tree          ✓")

        rf = self._rf()
        rf.fit(X_train, y_train)
        log.info("  [3/4] Random Forest          ✓")

        xgb = self._xgb(override=tuned_params)
        xgb.fit(X_train, y_train)
        log.info("  [4/4] XGBoost               ✓")

        self.models = {
            "Logistic Regression": lr,
            "Decision Tree":       dt,
            "Random Forest":       rf,
            "XGBoost":             xgb,
        }
        return self.models
