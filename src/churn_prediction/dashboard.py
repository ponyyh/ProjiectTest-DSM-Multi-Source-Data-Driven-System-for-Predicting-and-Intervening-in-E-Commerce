from pathlib import Path
import json
import pandas as pd
import streamlit as st

try:
    from .settings import PROJECT_ROOT
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from churn_prediction.settings import PROJECT_ROOT

st.set_page_config(page_title="用户流失预测看板", page_icon="📉", layout="wide")
st.title("📉 电商用户流失预测与干预系统")
reports = PROJECT_ROOT / "reports"; metrics_path = reports / "metrics.json"; importance_path = reports / "feature_importance.csv"
if not metrics_path.exists():
    st.warning("尚未找到训练结果，请先运行：python scripts/run_pipeline.py"); st.stop()
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
cols = st.columns(4); cols[0].metric("ROC-AUC", f"{metrics['auc_roc']:.3f}"); cols[1].metric("Precision", f"{metrics['precision']:.3f}"); cols[2].metric("Recall", f"{metrics['recall']:.3f}"); cols[3].metric("F1-Score", f"{metrics['f1']:.3f}")
left, right = st.columns(2)
with left:
    if (reports / "roc_curve.png").exists(): st.image(str(reports / "roc_curve.png"), caption="ROC 曲线")
with right:
    if (reports / "confusion_matrix.png").exists(): st.image(str(reports / "confusion_matrix.png"), caption="混淆矩阵")
if importance_path.exists():
    st.subheader("关键特征重要性"); st.bar_chart(pd.read_csv(importance_path).head(15).set_index("feature")["importance"])
st.caption("示例数据仅用于演示，生产环境请替换为经过授权的真实数据，并重新校准风险阈值。")
