# 电商用户流失预测与干预系统

一个面向电商业务的端到端数据科学项目示例，覆盖数据生成/接入、特征工程、模型训练、评估、SHAP 解释、FastAPI 推理接口和 Streamlit 看板。

## 快速开始

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .

python scripts/generate_sample_data.py
python scripts/run_pipeline.py
uvicorn churn_prediction.api:app --reload
streamlit run src/churn_prediction/dashboard.py
```

训练完成后会生成 `models/churn_pipeline.joblib`、`reports/metrics.json`、`reports/feature_importance.csv` 以及评估图表。

项目遵循 Cookiecutter Data Science 目录规范，并预留 DVC 数据版本控制位置。接入真实 MySQL/ERP 时，将数据落盘到 `data/raw/`，保持示例 CSV 的核心字段即可。

## 目录结构

```text
configs/       配置
data/          raw/interim/processed/external 数据
models/        序列化模型
reports/       指标、图表和解释结果
scripts/       数据生成和流水线入口
src/           业务代码
tests/         单元测试
```

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"user_id\":\"demo-001\",\"recency_days\":3,\"frequency_90d\":5,\"monetary_90d\":399,\"events_30d\":18,\"events_90d\":44,\"clicks_30d\":8,\"browses_30d\":7,\"carts_30d\":3,\"purchases_30d\":1,\"active_days_30d\":6,\"avg_order_amount_90d\":399,\"days_since_register\":120,\"city\":\"Shanghai\",\"channel\":\"organic\"}"
```

生产化建议：使用 DVC 管理数据和模型版本；按时间切分训练/测试集；根据业务成本矩阵校准风险阈值；为 API 增加鉴权、限流、日志和监控。
