# Food Access Prediction App

A full-stack application for predicting whether a census tract is classified as Low Income & Low Access (LILA) using a trained machine learning model.

## Project Structure

```text
.
├── backend/              # FastAPI application and model files
├── frontend/             # Streamlit user interface
└── docker-compose.yml    # Multi-container configuration
```

## Requirements

- Python 3.10+
- `pip`
- Docker and Docker Compose (optional)
- A trained model file (`model.pkl`)
- A preprocessing pipeline (`preprocessing.pkl`)

## Setup

Place the trained model and preprocessing files in the `backend/` directory:

```text
backend/
├── app.py
├── model.pkl
├── preprocessing.pkl
└── requirements.txt
```

Update the feature names and feature order in `backend/app.py` and `frontend/app.py` so they match the model used during training.

## Run Locally

### Backend

From the project root:

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The FastAPI documentation will be available at:

```text
http://localhost:8000/docs
```

### Frontend

Open a second terminal and run:

```bash
cd frontend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The Streamlit interface will be available at:

```text
http://localhost:8501
```

## Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

Open the application at:

```text
http://localhost:8501
```

Within the Docker network, the frontend connects to the backend at:

```text
http://backend:8000
```

To stop the containers:

```bash
docker compose down
```

## Configuration

Before running the application with a different model:

1. Update the feature list and input order in the backend.
2. Update the corresponding Streamlit input fields.
3. Confirm that preprocessing matches the pipeline used during model training.
4. Add required environment variables for the deployment environment.

## Deployment

The application can be deployed using its Docker configuration to services such as AWS, Google Cloud, or Microsoft Azure.

For production deployment, consider adding:

- Authentication and authorization
- CORS configuration
- Application logging
- Environment-based configuration
- Health checks
- Monitoring
