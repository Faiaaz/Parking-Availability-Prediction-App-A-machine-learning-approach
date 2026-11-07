"""Train classifiers + regressors on real Birmingham occupancy labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from config import (
    FEATURES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    OCCUPANCY_FULL_THRESHOLD,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_FRACTION,
)
from data_prep import (
    CALENDAR_FEATURES,
    FEATURE_COLUMNS,
    META_COLUMNS,
    TARGET_CLASS,
    TARGET_REG,
    build_features,
)


def load_feature_frame(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=["date_time"])
    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback, parse_dates=["date_time"])
    return build_features()


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Hold out the most recent portion of time — no random shuffle leakage."""
    ordered = df.sort_values("date_time").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    return train, test


def make_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["parking_zone"],
            ),
        ]
    )


def classification_models(numeric_cols: list[str]) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=16,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(
                            class_weight="balanced",
                            max_iter=5000,
                            dual="auto",
                            random_state=RANDOM_STATE,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def regression_models(numeric_cols: list[str]) -> dict:
    return {
        "ridge": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_reg": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=16,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_reg": Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def eval_classifier(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "task": "classification",
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "y_true": y_test.astype(int).tolist(),
        "y_proba": proba.astype(float).tolist(),
    }


def eval_regressor(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    pred = np.clip(pred, 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "model": name,
        "task": "regression",
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, pred)),
        "y_true": y_test.astype(float).tolist(),
        "y_pred": pred.astype(float).tolist(),
    }


def feature_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return df[numeric_cols + ["parking_zone"]]


def persistence_baselines(test: pd.DataFrame) -> dict:
    """Honest reference: last reading predicts the next."""
    y_c = test[TARGET_CLASS]
    y_r = test[TARGET_REG]
    lag = test["occ_rate_lag1"]
    pred_c = (lag < OCCUPANCY_FULL_THRESHOLD).astype(int)
    # Soft score for AUC: how far below the full threshold the lag sits.
    proba = np.clip(1.0 - lag, 0.0, 1.0)
    return {
        "classification": {
            "model": "persistence_lag1",
            "task": "classification",
            "accuracy": float(accuracy_score(y_c, pred_c)),
            "precision": float(precision_score(y_c, pred_c, zero_division=0)),
            "recall": float(recall_score(y_c, pred_c, zero_division=0)),
            "f1": float(f1_score(y_c, pred_c, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_c, proba)),
            "average_precision": float(average_precision_score(y_c, proba)),
            "confusion_matrix": confusion_matrix(y_c, pred_c).tolist(),
        },
        "regression": {
            "model": "persistence_lag1",
            "task": "regression",
            "mae": float(mean_absolute_error(y_r, lag)),
            "rmse": float(np.sqrt(mean_squared_error(y_r, lag))),
            "r2": float(r2_score(y_r, lag)),
        },
    }


def run_model_suite(
    name_prefix: str,
    numeric_cols: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
) -> tuple:
    X_train = feature_matrix(train, numeric_cols)
    X_test = feature_matrix(test, numeric_cols)
    y_train_c = train[TARGET_CLASS]
    y_test_c = test[TARGET_CLASS]
    y_train_r = train[TARGET_REG]
    y_test_r = test[TARGET_REG]

    best_auc, best_cls_name, best_cls_model = -1.0, None, None
    for name, model in classification_models(numeric_cols).items():
        key = f"{name_prefix}{name}" if name_prefix else name
        print(f"\nTraining classifier: {key}")
        model.fit(X_train, y_train_c)
        result = eval_classifier(key, model, X_test, y_test_c)
        curves = {"y_true": result.pop("y_true"), "y_proba": result.pop("y_proba")}
        result["feature_set"] = "calendar+lags" if not name_prefix else "calendar_only"
        metrics["classification"][key] = result
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")
        joblib.dump(curves, REPORTS_DIR / f"{key}_curves.joblib")
        print(
            f"  AUC={result['roc_auc']:.4f}  F1={result['f1']:.4f}  "
            f"Acc={result['accuracy']:.4f}"
        )
        if result["roc_auc"] > best_auc:
            best_auc = result["roc_auc"]
            best_cls_name = key
            best_cls_model = model

    best_r2, best_reg_name, best_reg_model = -1e9, None, None
    for name, model in regression_models(numeric_cols).items():
        key = f"{name_prefix}{name}" if name_prefix else name
        print(f"\nTraining regressor: {key}")
        model.fit(X_train, y_train_r)
        result = eval_regressor(key, model, X_test, y_test_r)
        curves = {"y_true": result.pop("y_true"), "y_pred": result.pop("y_pred")}
        result["feature_set"] = "calendar+lags" if not name_prefix else "calendar_only"
        metrics["regression"][key] = result
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")
        joblib.dump(curves, REPORTS_DIR / f"{key}_curves.joblib")
        print(
            f"  R²={result['r2']:.4f}  MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}"
        )
        if result["r2"] > best_r2:
            best_r2 = result["r2"]
            best_reg_name = key
            best_reg_model = model

    return best_cls_name, best_cls_model, best_reg_name, best_reg_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train parking availability models")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_frame(args.features)
    train, test = chronological_split(df)

    print(
        f"Train: {len(train):,} rows "
        f"({train['date_time'].min()} → {train['date_time'].max()})"
    )
    print(
        f"Test:  {len(test):,} rows "
        f"({test['date_time'].min()} → {test['date_time'].max()})"
    )
    print(
        f"Class balance — train available={train[TARGET_CLASS].mean():.3f}, "
        f"test available={test[TARGET_CLASS].mean():.3f}"
    )

    metrics: dict = {
        "dataset": "UCI Parking Birmingham",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": str(train["date_time"].min()),
        "train_end": str(train["date_time"].max()),
        "test_start": str(test["date_time"].min()),
        "test_end": str(test["date_time"].max()),
        "label_definition": (
            f"available = 1 if occupancy/capacity < {OCCUPANCY_FULL_THRESHOLD}"
        ),
        "split": "chronological hold-out (last 20%)",
        "notes": (
            "High short-horizon scores are expected because occupancy is persistent; "
            "compare against persistence_lag1 and calendar_only_* baselines."
        ),
        "classification": {},
        "regression": {},
        "baselines": {},
    }

    print("\n=== Feature set: calendar + lags (short-horizon forecast) ===")
    best_cls, best_cls_model, best_reg, best_reg_model = run_model_suite(
        "", FEATURE_COLUMNS, train, test, metrics
    )

    print("\n=== Feature set: calendar + zone only (no recent occupancy) ===")
    cal_cls, _, cal_reg, _ = run_model_suite(
        "calendar_only_", CALENDAR_FEATURES, train, test, metrics
    )

    baselines = persistence_baselines(test)
    metrics["baselines"] = baselines
    print("\n=== Persistence baseline (lag-1) ===")
    print(
        f"  clf AUC={baselines['classification']['roc_auc']:.4f}  "
        f"F1={baselines['classification']['f1']:.4f}"
    )
    print(
        f"  reg R²={baselines['regression']['r2']:.4f}  "
        f"MAE={baselines['regression']['mae']:.4f}"
    )

    holdout = test[META_COLUMNS + FEATURE_COLUMNS + [TARGET_CLASS, TARGET_REG]].copy()
    holdout.to_csv(REPORTS_DIR / "holdout_sample.csv", index=False)

    bundle = {
        "best_classifier": best_cls,
        "best_regressor": best_reg,
        "calendar_only_classifier": cal_cls,
        "calendar_only_regressor": cal_reg,
        "feature_columns": FEATURE_COLUMNS,
        "calendar_features": CALENDAR_FEATURES,
    }
    joblib.dump(
        {
            "classifier": best_cls_model,
            "regressor": best_reg_model,
            "meta": bundle,
        },
        MODELS_DIR / "best_bundle.joblib",
    )

    metrics["best_classifier"] = best_cls
    metrics["best_regressor"] = best_reg
    metrics["best_calendar_only_classifier"] = cal_cls
    metrics["best_calendar_only_regressor"] = cal_reg

    # Drop bulky nested report from JSON (already in joblib curves).
    slim = json.loads(json.dumps(metrics))
    for block in ("classification", "regression"):
        for model_name, payload in slim[block].items():
            payload.pop("classification_report", None)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)

    print(f"\nBest short-horizon classifier: {best_cls}")
    print(f"Best short-horizon regressor:  {best_reg}")
    print(f"Best calendar-only classifier: {cal_cls}")
    print(f"Metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
