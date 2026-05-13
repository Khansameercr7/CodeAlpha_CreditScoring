"""
Optuna hyperparameter search for XGBoost
"""
from __future__ import annotations
import optuna
from xgboost             import XGBClassifier
from sklearn.model_selection import cross_val_score
from src.utils.logger    import get_logger

log = get_logger("models.tuner")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Bayesian hyperparameter search using Optuna TPE sampler.

    Optuna builds a probabilistic model of the objective function
    and uses it to propose promising parameter regions — far more
    efficient than Grid Search or Random Search.
    """

    def __init__(self, n_trials=40, timeout=180, random_seed=42, cv_folds=5):
        self.n_trials    = n_trials
        self.timeout     = timeout
        self.random_seed = random_seed
        self.cv_folds    = cv_folds
        self.best_params_: dict = {}

    def _objective(self, trial, X, y, pos_weight):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 600),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            gamma             = trial.suggest_float("gamma", 0, 5),
            reg_alpha         = trial.suggest_float("reg_alpha", 0, 5),
            reg_lambda        = trial.suggest_float("reg_lambda", 0, 5),
            scale_pos_weight  = pos_weight,
        )
        model = XGBClassifier(**params, eval_metric="logloss",
                              random_state=self.random_seed, verbosity=0)
        scores = cross_val_score(model, X, y, cv=self.cv_folds,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    def tune(self, splits: dict) -> dict:
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        log.info(f"Optuna tuning XGBoost  "
                 f"(trials={self.n_trials}, timeout={self.timeout}s) …")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(
            lambda t: self._objective(t, X_train, y_train, pos_weight),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        self.best_params_ = study.best_params
        log.info(f"Best ROC-AUC (CV) : {study.best_value:.4f}")
        log.info(f"Best params       : {self.best_params_}")
        return self.best_params_
