from __future__ import annotations

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURE_COLUMNS, validate_features


def make_pipeline(random_state=42, model_type="tree"):
    numeric = [c for c in FEATURE_COLUMNS if c not in {"city", "channel"}]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["city", "channel"]),
    ])
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    else:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=random_state)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(frame, output_dir, random_state=42, test_size=0.2):
    validate_features(frame)
    if frame["churned"].nunique() < 2:
        raise ValueError("标签 churned 必须同时包含 0 和 1")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURE_COLUMNS], frame["churned"], test_size=test_size, stratify=frame["churned"], random_state=random_state)
    candidates = {}
    for model_type in ("logistic", "tree"):
        candidate = make_pipeline(random_state, model_type); candidate.fit(x_train, y_train)
        candidate_probabilities = candidate.predict_proba(x_test)[:, 1]
        candidates[model_type] = (candidate, candidate_probabilities)
    model_type, (pipeline, probabilities) = max(candidates.items(), key=lambda item: roc_auc_score(y_test, item[1][1]))
    predictions = (probabilities >= 0.5).astype(int)
    comparison = [{"model": name, "auc_roc": float(roc_auc_score(y_test, values[1]))} for name, values in candidates.items()]
    pd.DataFrame(comparison).to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    metrics = {"model": pipeline.named_steps["model"].__class__.__name__, "selected_model_type": model_type, "rows": int(len(frame)), "positive_rate": float(frame.churned.mean()), "accuracy": float(accuracy_score(y_test, predictions)), "precision": float(precision_score(y_test, predictions, zero_division=0)), "recall": float(recall_score(y_test, predictions, zero_division=0)), "f1": float(f1_score(y_test, predictions, zero_division=0)), "auc_roc": float(roc_auc_score(y_test, probabilities)), "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(pipeline, output_dir.parent / "models" / "churn_pipeline.joblib")
    _save_plots(y_test, predictions, probabilities, output_dir)
    return metrics


def _save_plots(y_true, y_pred, probabilities, output_dir):
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("预测"); plt.ylabel("实际"); plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, probabilities):.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "roc_curve.png", dpi=150); plt.close()
