import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from churn_prediction.explain import feature_importance
from churn_prediction.features import build_features
from churn_prediction.io import ensure_parent, read_raw_data
from churn_prediction.modeling import train_and_evaluate
from churn_prediction.settings import load_config


def main():
    config = load_config(); users, events, orders = read_raw_data(config["paths"]["raw"])
    dataset = build_features(users, events, orders, config["observation_end"], config["label_window_days"])
    dataset.to_csv(ensure_parent(config["paths"]["processed"] / "features.csv"), index=False, encoding="utf-8-sig")
    metrics = train_and_evaluate(dataset, config["paths"]["reports"], config["random_state"], config["test_size"])
    feature_importance(config["paths"]["models"] / "churn_pipeline.joblib", dataset, config["paths"]["reports"] / "feature_importance.csv")
    print(f"Training complete. ROC-AUC={metrics['auc_roc']:.4f}")


if __name__ == "__main__": main()
