FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt && pip install --user uvicorn

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY main.py ./
COPY random_forest_model.pkl ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]