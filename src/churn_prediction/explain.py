from pathlib import Path
import joblib
import pandas as pd

from .features import FEATURE_COLUMNS


def feature_importance(model_path, frame, output_path):
    pipeline = joblib.load(model_path)
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0]) if hasattr(model, "coef_") else [0.0] * len(names)
    result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True); result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def shap_values(model_path, frame):
    import shap
    pipeline = joblib.load(model_path)
    transformed = pipeline.named_steps["preprocessor"].transform(frame[FEATURE_COLUMNS])
    return shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
