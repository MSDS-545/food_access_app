# Food Access Prediction App

This repository contains a full-stack application for predicting whether a census tract is Low Income & Low Access (LILA) using a machine learning model.

## Structure
- `backend/`: FastAPI application serving the model  
- `frontend/`: Streamlit application for user interaction  
- `docker-compose.yml`: to build and run both services  

## Setup & Run
1. Place your trained model file (`model.pkl`) and preprocessing pipeline (`preprocessing.pkl`) into the `backend/` folder.  
2. Adjust feature names/order in `backend/app.py` and `frontend/app.py`.  
3. Choose your run mode:

### 🏃 Run Locally (without Docker)
```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Frontend (in new terminal)
cd ../frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Then open your browser:

Backend docs: http://localhost:8000/docs

Frontend UI: http://localhost:8501

🐳 Run with Docker / Docker Compose

From project root:

docker compose up --build


After build completes, open your browser:

http://localhost:8501

The frontend will automatically connect to the backend service at http://backend:8000.

Customization

Update the feature list and UI inputs as per your model.

Add environment variables, authentication, CORS, logging as needed.

Deploy to cloud (e.g., AWS, GCP, Azure) by pushing Docker images.