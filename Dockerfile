FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY reports ./reports
EXPOSE 8000
CMD ["uvicorn", "churn_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
