# F1 Podium Predictor - Setup & Installation Guide

Complete step-by-step guide to set up and run the full application.

---

## Prerequisites

- Python 3.9+
- Node.js 16+ (for React)
- Two terminal windows (one for backend, one for frontend)

---

## Phase 1: Backend Setup (Flask Server)

### Step 1: Create Backend Directory Structure
```
F1 ML/
├── backend/
│   ├── app.py              (created)
│   ├── requirements.txt    (created)
│   ├── driver_lookup.pkl   (from notebook)
│   ├── constructor_lookup.pkl (from notebook)
│   ├── circuit_lookup.pkl  (from notebook)
│   ├── randomforest_f1_model.pkl (from notebook)
│   └── catboost_f1_model.cbm (from notebook)
```

### Step 2: Copy/Move Model Files to Backend Directory

From your notebook output, copy these generated files to `backend/` folder:
- `driver_lookup.pkl`
- `constructor_lookup.pkl`
- `circuit_lookup.pkl`
- `randomforest_f1_model.pkl`
- `catboost_f1_model.cbm`
- `rf_feature_names.pkl`

### Step 3: Install Backend Dependencies

```bash
cd "path/to/F1 ML/backend"
pip install -r requirements.txt
```

### Step 4: Start Backend Server

```bash
python app.py
```

**Expected output:**
```
============================================================
F1 ML PREDICTION BACKEND
============================================================
Server running on http://localhost:5000
Available endpoints:
  GET  /api/drivers          - List all drivers
  GET  /api/constructors     - List all constructors
  GET  /api/circuits         - List all circuits
  POST /api/predict          - Make prediction
  GET  /api/health           - Health check
============================================================
```

**Keep this terminal open** - backend must run continuously.

---

## Phase 2: Frontend Setup (React App)

### Step 1: Create Frontend Directory Structure
```
F1 ML/
├── frontend/
│   ├── src/
│   │   ├── App.js          (created)
│   │   ├── App.css         (created)
│   │   ├── index.js        (created)
│   │   └── index.css       (created)
│   ├── public/
│   │   └── index.html      (created)
│   └── package.json        (created)
```

### Step 2: Install Frontend Dependencies

```bash
cd "path/to/F1 ML/frontend"
npm install
```

### Step 3: Start Frontend Server

```bash
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view f1-podium-predictor in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

**Open browser to** `http://localhost:3000`

---

## Phase 3: Test the App

### Test Backend Health
```bash
curl http://localhost:5000/api/health
```

Should return:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "lookups_loaded": true
}
```

### Test Full Prediction Flow

1. **In browser** (`http://localhost:3000`), you should see:
   - F1 Podium Predictor header
   - 4 input fields (Driver, Constructor, Circuit, Grid Position slider)
   - PREDICT PODIUM button

2. **Select inputs:**
   - Choose any Driver from dropdown
   - Choose any Constructor from dropdown
   - Choose any Circuit from dropdown
   - Set Grid Position (1-20)

3. **Click "PREDICT PODIUM"**

4. **View results:**
   - Should see prediction panel appear
   - Shows podium percentage from both models (RF & CatBoost)
   - Shows ensemble prediction
   - Shows final verdict (PODIUM / NO PODIUM)

---

## Troubleshooting

### Backend won't start
**Error:** `ModuleNotFoundError: No module named 'flask'`
- Solution: Make sure you ran `pip install -r requirements.txt` in the backend folder

**Error:** `FileNotFoundError: driver_lookup.pkl`
- Solution: Copy lookup table files from notebook output to backend directory

### Frontend won't connect to backend
**Error:** `Failed to load dropdown data` in browser console
- Check backend is running: `curl http://localhost:5000/api/health`
- If not running, start it: `python app.py` in backend terminal
- If CORS error: Backend already has CORS enabled (Flask-CORS)

### Models not loading
**Error:** `Models not loaded` when making prediction
- Verify both `.pkl` and `.cbm` files exist in backend directory
- Check file names match exactly in `app.py`

### Port conflicts
**Error:** `Port 5000 already in use` or `Port 3000 already in use`
- Backend: Edit `app.py` line, change `port=5000` to `port=5001`
- Frontend: Run `PORT=3001 npm start`
- Update `API_BASE` in `frontend/src/App.js` to match new backend URL

---

## Architecture Summary

```
User (http://localhost:3000)
    |
    └─ React App
         |
         ├─ GET /api/drivers → Load driver dropdown
         ├─ GET /api/constructors → Load constructor dropdown
         ├─ GET /api/circuits → Load circuit dropdown
         |
         └─ User selects inputs + clicks "PREDICT"
            |
            └─ POST /api/predict {driver_id, constructor_id, circuit_id, grid_position}
               |
               Flask Backend (http://localhost:5000)
               |
               ├─ Load driver stats from driver_lookup.pkl
               ├─ Load constructor stats from constructor_lookup.pkl
               ├─ Load circuit stats from circuit_lookup.pkl
               |
               ├─ Combine into feature vector
               ├─ Pass to RandomForestClassifier
               ├─ Pass to CatBoostClassifier
               |
               ├─ Average both predictions
               └─ Return: {podium_chance: "78.5%", prediction: "PODIUM"}
                  |
               React displays result
```

---

## Production Deployment

### Deploy Backend (Flask)
Options:
- **Heroku**: `git push heroku main`
- **AWS Lambda**: Serverless Flask with Zappa
- **Railway.app**: Simple cloud deployment
- **DigitalOcean App Platform**: Docker-based deployment

### Deploy Frontend (React)
Options:
- **Vercel**: `npm install -g vercel` then `vercel` in frontend directory
- **Netlify**: Connect GitHub repo to Netlify
- **Cloudflare Pages**: GitHub integration
- **AWS S3 + CloudFront**: Static hosting

**Important:** Update `API_BASE` in `frontend/src/App.js` to your production backend URL

---

## Files Summary

| File | Purpose | Created |
|------|---------|---------|
| `backend/app.py` | Flask server, model loading, prediction logic | Phase 1 |
| `backend/requirements.txt` | Python dependencies | Phase 1 |
| `frontend/src/App.js` | React main component, UI logic | Phase 2 |
| `frontend/src/App.css` | UI styling (F1 themed) | Phase 2 |
| `frontend/src/index.js` | React entry point | Phase 2 |
| `frontend/package.json` | Node dependencies, npm scripts | Phase 2 |
| `frontend/public/index.html` | HTML shell | Phase 2 |

---

## Done! 🏎️

Your full F1 ML prediction app is ready. Users can now:
1. Select driver, constructor, circuit via dropdowns
2. Adjust grid position with slider
3. Get instant ML prediction with probability percentages
4. See breakdown from 2 models + ensemble result
