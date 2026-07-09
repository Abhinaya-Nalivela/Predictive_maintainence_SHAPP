from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_RECALL =  0.95
N_SPLITS = 5
N_ITER_SEARCH = 50

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODELS_DIR = BASE_DIR / "models"

FINAL_FEATURES = [
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


def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}\n"
            f"Expected location: {DATA_PATH}"
        )
    return pd.read_csv(path)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def preprocess(df):
    df = clean_columns(df)

    required_cols = [
        "UDI",
        "Product_ID",
        "Type",
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
        "Machine_failure",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop(columns=["UDI", "Product_ID"])
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    if df["Type"].isna().any():
        raise ValueError("Unexpected category found in Type column.")

    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_proxy"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
    df["Wear_speed_ratio"] = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1)
    df["Torque_wear_interaction"] = df["Torque_Nm"] * df["Tool_wear_min"]
    df["Temp_torque_interaction"] = df["Temp_diff_K"] * df["Torque_Nm"]
    df["Wear_temp_interaction"] = df["Tool_wear_min"] * df["Temp_diff_K"]

    return df


def build_threshold_table(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    rows = []
    for i, thr in enumerate(thresholds):
        p = float(precision[i + 1])
        r = float(recall[i + 1])
        f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        rows.append({
            "threshold": float(thr),
            "precision": p,
            "recall": r,
            "f1": float(f1),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
        })

    return pd.DataFrame(rows)


def select_threshold(y_true, y_prob, min_recall=TARGET_RECALL):
    threshold_table = build_threshold_table(y_true, y_prob)
    eligible = threshold_table[threshold_table["recall"] >= min_recall].copy()

    if not eligible.empty:
        eligible = eligible.sort_values(
            by=["precision", "f1", "threshold"],
            ascending=[False, False, False],
        )
        best = eligible.iloc[0].to_dict()
        rule = (
            f"Selected highest-precision threshold among candidates with "
            f"recall >= {min_recall:.4f}; tie-break by F1, then threshold."
        )
        return best, threshold_table, rule

    fallback = threshold_table.sort_values(
        by=["recall", "precision", "f1"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    rule = (
        f"No threshold achieved recall >= {min_recall:.4f}; "
        f"selected fallback threshold with maximum recall, then precision, then F1."
    )
    return fallback, threshold_table, rule


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
    }, y_pred


def main():
    df = preprocess(load_dataset())

    X = df[FINAL_FEATURES]
    y = df["Machine_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    class_ratio = neg / pos

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Positive failures in training:", pos)
    print("Negative cases in training:", neg)
    print(f"scale_pos_weight base ratio: {class_ratio:.4f}")

    model = XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 2, 3, 5],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
        "max_delta_step": [0, 1, 3],
        "scale_pos_weight": [
            class_ratio * 0.75,
            class_ratio * 1.0,
            class_ratio * 1.25,
            class_ratio * 1.5,
        ],
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    print("\nStarting randomized search...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"\nBest CV Recall: {search.best_score_:.6f}")
    print("Best Params:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")

    print("\nGenerating out-of-fold probabilities for threshold selection...")
    oof_prob = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_info, threshold_table, threshold_rule = select_threshold(
        y_train,
        oof_prob,
        min_recall=TARGET_RECALL,
    )
    best_threshold = float(threshold_info["threshold"])

    print(f"\nSelected threshold: {best_threshold:.6f}")
    print("Threshold rule:", threshold_rule)
    print("Threshold metrics:", threshold_info)

    print("\nFitting final model on full training data...")
    best_model.fit(X_train, y_train)

    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics, y_test_pred = evaluate_at_threshold(y_test, y_test_prob, best_threshold)

    test_ap = average_precision_score(y_test, y_test_prob)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Threshold: {best_threshold:.6f}")
    print(f"Accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Precision: {test_metrics['precision']:.6f}")
    print(f"Recall: {test_metrics['recall']:.6f}")
    print(f"F1-score: {test_metrics['f1']:.6f}")
    print(f"PR-AUC: {test_ap:.6f}")
    print(f"ROC-AUC: {test_roc_auc:.6f}")
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgb_model.joblib"
    meta_path_joblib = MODELS_DIR / "model_meta.joblib"
    meta_path_json = MODELS_DIR / "model_meta.json"
    threshold_csv_path = MODELS_DIR / "threshold_candidates.csv"

    joblib.dump(best_model, model_path)

    model_meta = {
        "features": FINAL_FEATURES,
        "best_threshold": best_threshold,
        "best_params": search.best_params_,
        "cv_recall_score": float(search.best_score_),
        "test_average_precision": float(test_ap),
        "test_roc_auc": float(test_roc_auc),
        "final_accuracy": float(test_metrics["accuracy"]),
        "final_precision": float(test_metrics["precision"]),
        "final_recall": float(test_metrics["recall"]),
        "final_f1": float(test_metrics["f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "threshold_selection_rule": threshold_rule,
        "threshold_details": threshold_info,
    }

    joblib.dump(model_meta, meta_path_joblib)

    with open(meta_path_json, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    threshold_table.to_csv(threshold_csv_path, index=False)

    print("\nSaved:")
    print(f"Model -> {model_path}")
    print(f"Metadata -> {meta_path_joblib}")
    print(f"Metadata JSON -> {meta_path_json}")
    print(f"Threshold table -> {threshold_csv_path}")


if __name__ == "__main__":
    main()