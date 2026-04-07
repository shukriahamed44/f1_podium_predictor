# F1 Podium Predictor

## Overview

The F1 Podium Predictor is an end-to-end Machine Learning pipeline and web application designed to forecast podium finishes for Formula 1 races. Utilizing historical performance metrics and circuit data spanning from 2020 to 2024, the system leverages a robust ensemble model combining Random Forest and CatBoost classifiers to generate empirical win probabilities for all 22 grid positions.

The application is decoupled into three primary segments: an offline model training pipeline (Jupyter Notebook), an inference-ready REST API backend (Flask), and a responsive client interface (React).

## Architecture

- **Offline ML Pipeline (`GP26_ML_Project.ipynb`)**: Responsible for raw data cleaning, handling missing time-series metrics via backward/forward filling, dataset normalization, and offline model training. Post-analysis, it exports serialized model artifacts (`.pkl`, `.cbm`) alongside highly optimized Pandas lookup tables to bypass live database polling constraints during inference.
- **Backend API (`/backend`)**: A lightweight Python Flask server. It ingests the serialized models and lookup tables into RAM on startup, enabling latency-optimized requests. It exposes RESTful endpoints (e.g., `/api/predict-batch`) to compute and serve high-throughput ensemble predictions.
- **Frontend Interface (`/frontend`)**: A React.js single-page application orchestrating the client experience. It retrieves runtime data grids asynchronously, submits normalized HTTP payloads to the backend, and displays real-time ranked probabilistic verdicts for every grid position dynamically. 

## ML Pipeline Mechanics

1. **Feature Engineering**: Native telemetry and race metrics are synthesized into critical interaction features directly impacting race pace (e.g., `grid_x_driver_consistency`). 
2. **Data Leakage Mitigation**: Obvious post-race features (accumulated points, laps completed, fastest lap indices) are systematically purged early in the training array to guarantee models rely purely on pre-race intel.
3. **Imbalanced Class Handling**: As podium finishes represent a heavy minority class, the classification algorithms enforce balanced class weights and strictly tune hyperparameters against the F1-Score (via GridSearchCV) rather than simple Accuracy.
4. **Input Translation**: The live environment shields users from strict ML schema limits. Incoming user payloads require minimal native identifiers (driver, constructor, circuit indices) while the backend silently retrieves and scales complex parameters (historical win percentages) before passing the vector matrix to the ensemble models.

## Local Setup & Installation

Follow these steps to deploy and test the application on local system contexts. 

### Prerequisites
- Python 3.9+
- Node.js 16+

### 1. Backend Service (Flask)
Boot up the backend predictive API. The environment requires the generated `.pkl` and `.cbm` model artifacts within the root of its boundary to spin up successfully.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The Flask server will mount at `http://localhost:5000` and locally cache the model graphs into available memory. Ensure this subsystem remains active to serve requests.

### 2. Frontend Application (React)
Initialize a separate terminal session to orchestrate the client interface.

```bash
cd frontend
npm install
npm start
```
The React development server will start locally at `http://localhost:3000`. The frontend maps endpoints to handle Cross-Origin Resource Sharing (CORS) natively with the backend structure. 

## Production Deployment Environment

The repository supports native transitions to standard continuous deployment clouds:

- **Backend Layers**: Best processed via encapsulated Docker containers, or directly deployed onto optimized Python-serving environments such as Heroku, AWS (Elastic Beanstalk/Lambda via Zappa), or Railway.
- **Frontend Client**: Build properties are fully compatible with Vercel, Netlify, or Cloudflare platforms for edge network hosting. 

For deployment transitions, ensure the hardcoded local endpoint mapping variable in `frontend/src/App.js` transitions to the live remote domain URI prior to the finalized UI build step.
