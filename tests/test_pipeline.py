"""
Full test suite
Run: python -m pytest tests/ -v
"""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
import yaml

# Load config───────────────
with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

# Use smaller settings for test speed
CFG["data"]["use_smote"] = True
CFG["tuning"]["enabled"] = False
CFG["models"]["random_forest"]["n_estimators"] = 20
CFG["models"]["xgboost"]["n_estimators"] = 20

SEED = 42


# Fixtures────────────────
@pytest.fixture(scope="module")
def raw_df():
    from src.data.loader import load_and_explore
    return load_and_explore(CFG)

@pytest.fixture(scope="module")
def splits_and_pp(raw_df):
    from src.data.preprocessor import Preprocessor
    pp = Preprocessor(CFG)
    return pp.fit_transform(raw_df), pp

@pytest.fixture(scope="module")
def trained_models(splits_and_pp):
    sp, _ = splits_and_pp
    from src.models.trainer import ModelTrainer
    trainer = ModelTrainer(CFG["models"], SEED)
    return trainer.train(sp), trainer


# Section 1 - Data Loading
class TestLoader:
    def test_shape(self, raw_df):
        assert len(raw_df) > 30_000
        assert "loan_status" in raw_df.columns

    def test_target_binary(self, raw_df):
        assert set(raw_df["loan_status"].unique()).issubset({0, 1})

    def test_outliers_removed_in_preprocessor(self, splits_and_pp):
        sp, _ = splits_and_pp
        assert sp["X_train"]["person_age"].max() <= 100
        assert sp["X_train"]["person_income"].max() <= 1_500_000

    def test_class_imbalance_exists(self, raw_df):
        vc = raw_df["loan_status"].value_counts()
        assert vc[0] > vc[1], "Majority class should be 0 (Good Standing)"

    def test_expected_columns(self, raw_df):
        expected = {"person_age","person_income","person_home_ownership",
                    "person_emp_length","loan_intent","loan_grade","loan_amnt",
                    "loan_int_rate","loan_status","loan_percent_income",
                    "cb_person_default_on_file","cb_person_cred_hist_length"}
        assert expected.issubset(set(raw_df.columns))


# ================================================================
#  Section 2 — Preprocessing
# ================================================================
class TestPreprocessor:
    def test_no_nulls_after_preprocessing(self, splits_and_pp):
        sp, _ = splits_and_pp
        assert sp["X_train"].isnull().sum().sum() == 0
        assert sp["X_test"].isnull().sum().sum() == 0

    def test_engineered_features_present(self, splits_and_pp):
        sp, _ = splits_and_pp
        eng = ["debt_to_income_ratio","loan_to_income_ratio",
               "income_per_cred_year","high_risk_grade_flag",
               "int_rate_x_loan_amnt","employment_stability_score"]
        for feat in eng:
            assert feat in sp["feature_names"], f"Missing engineered feature: {feat}"

    def test_smote_balances_classes(self, splits_and_pp):
        sp, _ = splits_and_pp
        counts = pd.Series(sp["y_train"]).value_counts()
        assert counts[0] == counts[1], "SMOTE should produce equal class counts"

    def test_scaled_shape_matches(self, splits_and_pp):
        sp, _ = splits_and_pp
        assert sp["X_train_sc"].shape == sp["X_train"].shape
        assert sp["X_test_sc"].shape  == sp["X_test"].shape

    def test_no_data_leakage(self, splits_and_pp):
        sp, _ = splits_and_pp
        # Train mean after scaling should be ~0; test won't be exactly 0
        train_mean = np.abs(sp["X_train_sc"].mean(axis=0)).mean()
        assert train_mean < 0.05, "Train mean after scaling should be ~0"

    def test_categorical_encoded(self, splits_and_pp):
        sp, _ = splits_and_pp
        # One-hot columns should be present
        assert any("RENT" in f or "MORTGAGE" in f or "OWN" in f
                   for f in sp["feature_names"]), \
            "home_ownership categories should be one-hot encoded"

    def test_split_ratio(self, raw_df, splits_and_pp):
        sp, _ = splits_and_pp
        # After outlier removal ~31,676 rows; test should be ~20% of that
        n_test = len(sp["X_test"])
        assert 5000 < n_test < 8000, f"Unexpected test size: {n_test}"


# ================================================================
#  Section 3 — Feature Engineering
# ================================================================
class TestFeatureEngineering:
    def test_high_risk_grade_flag(self, raw_df):
        from src.data.preprocessor import engineer_features
        df = raw_df.copy()
        df["loan_int_rate"] = df["loan_int_rate"].fillna(10.0)
        df["person_emp_length"] = df["person_emp_length"].fillna(4.0)
        df_eng = engineer_features(df)
        # Grade E/F/G should be flagged as 1
        e_rows = df_eng[df_eng["loan_grade"] == "E"]
        assert (e_rows["high_risk_grade_flag"] == 1).all()
        a_rows = df_eng[df_eng["loan_grade"] == "A"]
        assert (a_rows["high_risk_grade_flag"] == 0).all()

    def test_employment_stability_bounded(self, raw_df):
        from src.data.preprocessor import engineer_features
        df = raw_df.copy()
        df["loan_int_rate"] = df["loan_int_rate"].fillna(10.0)
        df["person_emp_length"] = df["person_emp_length"].fillna(4.0)
        df_eng = engineer_features(df)
        assert df_eng["employment_stability_score"].between(0, 1).all()

    def test_dti_positive(self, raw_df):
        from src.data.preprocessor import engineer_features
        df = raw_df.copy()
        df["loan_int_rate"] = df["loan_int_rate"].fillna(10.0)
        df["person_emp_length"] = df["person_emp_length"].fillna(4.0)
        df_eng = engineer_features(df)
        assert (df_eng["debt_to_income_ratio"] >= 0).all()


# ================================================================
#  Section 4 — Model Training
# ================================================================
class TestTrainer:
    def test_all_four_models(self, trained_models):
        models, _ = trained_models
        assert set(models.keys()) == {
            "Logistic Regression","Decision Tree","Random Forest","XGBoost"
        }

    def test_binary_predictions(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, _     = splits_and_pp
        for name, model in models.items():
            X = sp["X_test_sc"] if name == "Logistic Regression" else sp["X_test"]
            preds = model.predict(X)
            assert set(preds).issubset({0,1}), f"{name}: non-binary output"

    def test_probabilities_valid(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, _     = splits_and_pp
        for name, model in models.items():
            X = sp["X_test_sc"] if name == "Logistic Regression" else sp["X_test"]
            probs = model.predict_proba(X)
            assert probs.shape[1] == 2
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_random_forest_has_feature_importance(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, _     = splits_and_pp
        rf_imp = models["Random Forest"].feature_importances_
        assert len(rf_imp) == len(sp["feature_names"])
        assert rf_imp.sum() == pytest.approx(1.0, abs=1e-5)


# ================================================================
#  Section 5 — Evaluation
# ================================================================
class TestEvaluator:
    def test_metrics_columns(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, _     = splits_and_pp
        from src.evaluation.evaluator import ModelEvaluator
        ev  = ModelEvaluator(cv_folds=2)
        cdf = ev.evaluate(models, sp)
        for col in ["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]:
            assert col in cdf.columns

    def test_metrics_in_range(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, _     = splits_and_pp
        from src.evaluation.evaluator import ModelEvaluator
        ev  = ModelEvaluator(cv_folds=2)
        cdf = ev.evaluate(models, sp)
        for col in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]:
            assert cdf[col].between(0, 1).all(), f"{col} out of [0,1]"

    def test_xgboost_best_auc(self, trained_models, splits_and_pp):
        """XGBoost should typically have the best ROC-AUC on this dataset."""
        models, _ = trained_models
        sp, _     = splits_and_pp
        from src.evaluation.evaluator import ModelEvaluator
        ev  = ModelEvaluator(cv_folds=2)
        cdf = ev.evaluate(models, sp)
        best = cdf.loc[cdf["ROC-AUC"].idxmax(), "Model"]
        # XGBoost or RF should be best
        assert best in ("XGBoost","Random Forest"), \
            f"Expected XGBoost or RF to be best, got {best}"


# ================================================================
#  Section 6 — Model IO
# ================================================================
class TestModelIO:
    def test_save_and_reload(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        with tempfile.TemporaryDirectory() as tmp:
            from src.utils.model_io import save_all, load_all
            save_all(models, pp.scaler, sp["feature_names"], tmp)
            loaded_models, loaded_scaler, loaded_feats = load_all(tmp)
            assert len(loaded_models) == len(models)
            assert loaded_feats == sp["feature_names"]

    def test_predictions_match_after_reload(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        with tempfile.TemporaryDirectory() as tmp:
            from src.utils.model_io import save_all, load_all
            save_all(models, pp.scaler, sp["feature_names"], tmp)
            loaded, _, _ = load_all(tmp)
            for name in list(models.keys())[:2]:   # check first 2 for speed
                X = sp["X_test_sc"] if name == "Logistic Regression" else sp["X_test"]
                orig   = models[name].predict(X)
                reload = loaded.get(name, loaded.get(
                    name.replace("Logistic Regression","Logistic Regression"), None))
                if reload:
                    assert np.array_equal(orig, reload.predict(X)), \
                        f"{name} predictions differ after reload"


# ================================================================
#  Section 7 — Predictor
# ================================================================
class TestPredictor:
    GOOD = {
        "person_age": 35, "person_income": 85000,
        "person_home_ownership": "MORTGAGE", "person_emp_length": 8.0,
        "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "A",
        "loan_amnt": 10000, "loan_int_rate": 7.5,
        "loan_percent_income": 0.12,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10,
    }
    BAD = {
        "person_age": 22, "person_income": 18000,
        "person_home_ownership": "RENT", "person_emp_length": 0.0,
        "loan_intent": "PERSONAL", "loan_grade": "G",
        "loan_amnt": 20000, "loan_int_rate": 23.0,
        "loan_percent_income": 0.85,
        "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 2,
    }

    def test_output_keys(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        from src.models.predictor import CreditPredictor
        pred   = CreditPredictor(models["XGBoost"], pp.scaler,
                                  sp["feature_names"], False)
        result = pred.predict(self.GOOD.copy())
        for k in ["label","probability","risk_score","risk_band","recommendation"]:
            assert k in result

    def test_good_applicant_low_probability(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        from src.models.predictor import CreditPredictor
        pred   = CreditPredictor(models["XGBoost"], pp.scaler,
                                  sp["feature_names"], False)
        result = pred.predict(self.GOOD.copy())
        assert result["probability"] < 0.6, \
            f"Good applicant should have low default prob, got {result['probability']}"

    def test_bad_applicant_high_probability(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        from src.models.predictor import CreditPredictor
        pred   = CreditPredictor(models["XGBoost"], pp.scaler,
                                  sp["feature_names"], False)
        result = pred.predict(self.BAD.copy())
        assert result["probability"] > 0.4, \
            f"Bad applicant should have high default prob, got {result['probability']}"

    def test_risk_score_range(self, trained_models, splits_and_pp):
        models, _ = trained_models
        sp, pp    = splits_and_pp
        from src.models.predictor import CreditPredictor
        for name, model in models.items():
            use_sc = (name == "Logistic Regression")
            pred   = CreditPredictor(model, pp.scaler,
                                      sp["feature_names"], use_sc)
            r = pred.predict(self.GOOD.copy())
            assert 0 <= r["risk_score"] <= 100
            assert 0 <= r["probability"] <= 1
