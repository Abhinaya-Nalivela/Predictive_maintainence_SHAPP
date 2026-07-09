import os
import joblib
import kagglehub
import numpy as np
import pandas as pd

from kagglehub import KaggleDatasetAdapter
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
MIN_RECALL = 0.90
N_SPLITS = 5


def load_dataset():
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
        "ai4i2020.csv",
    )


def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = df.columns.str.replace("[", "", regex=False)
    df.columns = df.columns.str.replace("]", "", regex=False)
    df.columns = df.columns.str.replace("/", "_", regex=False)
    df = df.drop(["UDI", "Product_ID"], axis=1)
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_proxy"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
    df["Wear_speed_ratio"] = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1)
    df["Torque_wear_interaction"] = df["Torque_Nm"] * df["Tool_wear_min"]
    df["Temp_torque_interaction"] = df["Temp_diff_K"] * df["Torque_Nm"]
    df["Wear_temp_interaction"] = df["Tool_wear_min"] * df["Temp_diff_K"]
    return df


def threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": int(cm[0, 1]),
        "missed_failures": int(cm[1, 0]),
    }


def pick_threshold(y_true, y_prob, min_recall=MIN_RECALL, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    eligible = []
    fallback = None

    for threshold in thresholds:
        metrics = threshold_metrics(y_true, y_prob, threshold)

        if fallback is None or (
            metrics["recall"] > fallback["recall"]
            or (metrics["recall"] == fallback["recall"] and metrics["f1"] > fallback["f1"])
            or (
                metrics["recall"] == fallback["recall"]
                and metrics["f1"] == fallback["f1"]
                and metrics["precision"] > fallback["precision"]
            )
        ):
            fallback = metrics

        if metrics["recall"] >= min_recall:
            eligible.append(metrics)

    if eligible:
        eligible.sort(
            key=lambda m: (m["recall"], m["f1"], m["precision"]),
            reverse=True,
        )
        return eligible[0]

    return fallback


def cross_validated_threshold_search(base_model, X, y, cv, min_recall=MIN_RECALL):
    fold_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(base_model)
        model.fit(X_tr, y_tr)
        val_prob = model.predict_proba(X_val)[:, 1]
        best_fold = pick_threshold(y_val, val_prob, min_recall=min_recall)
        fold_thresholds.append(best_fold)

        print(
            f"Fold {fold_idx}: threshold={best_fold['threshold']:.2f}, "
            f"precision={best_fold['precision']:.3f}, recall={best_fold['recall']:.3f}, f1={best_fold['f1']:.3f}"
        )

    selected_threshold = float(np.median([m["threshold"] for m in fold_thresholds]))
    print(f"\nSelected threshold from CV median: {selected_threshold:.2f}")
    return selected_threshold, fold_thresholds


def main():
    df = preprocess(load_dataset())

    features = [
        "Type",
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
        "Temp_diff_K",
        "Power_proxy",
        "Wear_speed_ratio",
        "Torque_wear_interaction",
        "Temp_torque_interaction",
        "Wear_temp_interaction",
    ]

    X = df[features]
    y = df["Machine_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    base_scale_pos_weight = neg / pos

    print("Features used for training:", features)
    print(f"Class balance in training set -> negative: {neg}, positive: {pos}")
    print(f"Base scale_pos_weight from class ratio: {base_scale_pos_weight:.2f}")

    xgb = XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [300, 400, 500, 700, 900],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 2, 3, 5, 7, 10],
        "subsample": [0.75, 0.8, 0.85, 0.9, 1.0],
        "colsample_bytree": [0.65, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "max_delta_step": [0, 1, 3, 5, 7],
        "scale_pos_weight": [
            max(1, base_scale_pos_weight * 1.0),
            base_scale_pos_weight * 1.25,
            base_scale_pos_weight * 1.5,
            base_scale_pos_weight * 2.0,
            base_scale_pos_weight * 2.5,
            base_scale_pos_weight * 3.0,
            base_scale_pos_weight * 4.0,
        ],
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=100,
        scoring="recall",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    print("\nBest cross-validation score (Recall):")
    print(f"{random_search.best_score_:.4f}")

    print("\nBest parameters found:")
    for key, value in random_search.best_params_.items():
        print(f"{key}: {value}")

    best_threshold, fold_thresholds = cross_validated_threshold_search(
        best_model, X_train, y_train, cv, min_recall=MIN_RECALL
    )

    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    test_ap = average_precision_score(y_test, y_prob)
    test_roc_auc = roc_auc_score(y_test, y_prob)
    final_precision = precision_score(y_test, y_pred, zero_division=0)
    final_recall = recall_score(y_test, y_pred, zero_division=0)
    final_f1 = f1_score(y_test, y_pred, zero_division=0)
    final_cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest PR-AUC (Average Precision): {test_ap:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"\nFinal metrics at threshold = {best_threshold:.2f}")
    print(final_cm)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgb_model.joblib")

    model_meta = {
        "features": features,
        "best_threshold": float(best_threshold),
        "best_params": random_search.best_params_,
        "cv_recall_score": float(random_search.best_score_),
        "test_average_precision": float(test_ap),
        "test_roc_auc": float(test_roc_auc),
        "final_precision": float(final_precision),
        "final_recall": float(final_recall),
        "final_f1": float(final_f1),
        "confusion_matrix": final_cm.tolist(),
        "threshold_selection_rule": "CV median threshold maximizing recall with recall >= 0.90, tie-break by F1 and precision",
        "cv_fold_thresholds": fold_thresholds,
    }
    joblib.dump(model_meta, "models/model_meta.joblib")

    print("Model saved to models/xgb_model.joblib")
    print("Metadata saved to models/model_meta.joblib")
    print(f"Recommended dashboard threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()