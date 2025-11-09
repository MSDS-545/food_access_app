FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy multiple files into /app/ directory — note the trailing slash
COPY model.pkl preprocessing.pkl imputer_3features.pkl model_3features.pkl scaler_3features.pkl app.py /app/

EXPOSE 8000

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
#CMD ["uvicorn", "app:app", "--host", "$UVICORN_HOST", "--port", "$UVICORN_PORT"]


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

