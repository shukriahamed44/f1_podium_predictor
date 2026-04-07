# F1 ML Prediction UI - Plan & Architecture

## TL;DR: User Flow
**User picks 4 things → Backend looks up all stats → Model predicts → Shows % chance**

---

## Frontend UI Layer (React)

### User Inputs (What They See)
```
[Dropdown: Select Driver Nameㅤㅤㅤㅤ]
[Dropdown: Select Constructor Nameㅤㅤㅤㅤ]
[Dropdown: Select CircuitNameㅤㅤㅤㅤ]
[Slider: Grid Position (1-20)ㅤㅤㅤㅤ]
[Button: PREDICT PODIUM]
```

**3 Dropdowns, 1 Slider = All user needs to input**
- No model features, no IDs, no technical knowledge required
- Dropdowns pre-populated from `driver_lookup`, `constructor_lookup`, `circuit_lookup`

### Output Display
```
Max Verstappen @ Red Bull → Monaco, Grid 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PODIUM CHANCE:  78.5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Details:
  Random Forest:  76.2%
  CatBoost:       80.8%
  Ensemble:       78.5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prediction: PODIUM FINISH ✓
```

---

## Backend Layer (What Happens Behind the Scenes)

### Step 1: Load Lookup Tables (On App Start)
```
Backend reads from disk:
  - driver_lookup.pkl (all drivers + their stats)
  - constructor_lookup.pkl (all constructors + their stats)
  - circuit_lookup.pkl (all circuits + their track stats)
  - Trained RF model (.pkl)
  - Trained CatBoost model (.cbm)
```
**One time only**, then cached in memory.

### Step 2: Populate Dropdowns
```
React frontend fetches dropdown options from backend:
  GET /api/drivers → Returns list of driver names + IDs
  GET /api/constructors → Returns list of constructor names + IDs
  GET /api/circuits → Returns list of circuit names + IDs
```
These come from the loaded lookup tables.

### Step 3: User Submits Prediction Request
```
Frontend sends:
  POST /api/predict {
    driver_id: 1,
    constructor_id: 3,
    circuit_id: 6,
    grid_position: 3
  }
```

### Step 4: Backend Processes (The Lookup + Embed Magic)
```
1. Lookup step:
   - Find driver with ID=1 in driver_lookup → Extract: age, wins, podium %
   - Find constructor ID=3 → Extract: constructor top 3 %, wins
   - Find circuit ID=6 → Extract: track length, turns

2. Embed step:
   - Combine all fetched features + grid position into single feature vector
   - Apply same transformations as training (MinMaxScaler, interactions)

3. Predict step:
   - Feed to Random Forest model → Get probability
   - Feed to CatBoost model → Get probability
   - Average both → Final prediction
```

### Step 5: Backend Returns Result
```
Response to frontend:
  {
    "driver_id": 1,
    "constructor_id": 3,
    "circuit_id": 6,
    "grid_position": 3,
    "random_forest_podium_chance": "76.2%",
    "catboost_podium_chance": "80.8%",
    "ensemble_podium_chance": "78.5%",
    "prediction": "PODIUM"
  }
```

### Step 6: Frontend Renders Result
React displays the prediction with visual indicators (progress bar, color coding, etc.)

---

## Tech Stack

### Frontend
- **React**: Component-based UI (for dropdowns, sliders, results display)
- **Axios**: HTTP requests to backend API
- **Tailwind/Material-UI**: Styling (optional but recommended)

### Backend
- **Flask or FastAPI** (Python): Lightweight, perfect for ML serving
  - Loads models & lookup tables once at startup
  - Exposes REST endpoints
  - Handles lookup + predict logic
- **Joblib/Pickle**: Serialization for models & lookup tables

### Data Flow Diagram
```
React Frontend
    |
    └─→ [User selects driver/constructor/circuit/grid]
        |
        └─→ HTTP POST to Flask Backend
            |
            └─→ Backend:
                ├─ Lookup driver ID in driver_lookup
                ├─ Lookup constructor ID in constructor_lookup
                ├─ Lookup circuit ID in circuit_lookup
                ├─ Combine all features
                ├─ Apply transformations
                ├─ RF model.predict_proba()
                ├─ CatBoost model.predict_proba()
                ├─ Average results
                └─ Return JSON response
            |
            └─→ React receives {podium_chance: "78.5%", ...}
                |
                └─→ Display result to user
```

---

## API Endpoints Needed

### For Dropdown Population
```
GET /api/drivers
  Returns: [{id: 1, name: "Max Verstappen"}, {id: 2, name: "Lewis Hamilton"}, ...]

GET /api/constructors
  Returns: [{id: 1, name: "Red Bull"}, {id: 2, name: "Mercedes"}, ...]

GET /api/circuits
  Returns: [{id: 1, name: "Monaco"}, {id: 2, name: "Monza"}, ...]
```

### For Predictions
```
POST /api/predict
  Input: {driver_id: int, constructor_id: int, circuit_id: int, grid_position: int}
  Output: {
    random_forest_podium_chance: "76.2%",
    catboost_podium_chance: "80.8%",
    ensemble_podium_chance: "78.5%",
    prediction: "PODIUM"
  }
```

---

## Under the Hood: What Makes This Smart

**The Lookup-Embed Pattern**:
1. **No IDs leaked to model**: Model doesn't see raw IDs (which are just database keys)
2. **Automatic feature engineering**: When user inputs ID, system automatically fetches all derived features
3. **User-friendly**: Users think in terms of driver/team names, not technical feature vectors
4. **Production-ready**: Same pattern used in real ML APIs for tabular data prediction

**Why Ensemble?**
- Two models often disagree (RF says 76%, CatBoost says 81%)
- Averaging (78.5%) balances both approaches
- Shows uncertainty to user (if RF & CB differ widely, prediction is less confident)

---

## Deployment

### Option 1: Local (Development)
- Run Flask backend locally
- React dev server locally
- Both on localhost for testing

### Option 2: Cloud (Production)
- Backend: Deploy Flask to Heroku, AWS Lambda, or Railway
- Frontend: Deploy React to Vercel, Netlify, or Cloudflare Pages
- Models & lookup tables bundled with backend service

### File Size Considerations
- RF model: ~50MB (pickle format)
- CatBoost model: ~30MB
- Lookup tables: ~5MB
- Total: ~85MB (reasonable for any cloud platform)

---

## Summary

**What User Sees**: 4 simple inputs (dropdowns + slider) → prediction percentage

**What Backend Does**: 
1. Takes user's 4 inputs (IDs + grid)
2. Looks up all corresponding features from cached lookup tables
3. Combines features (embedding step)
4. Runs through 2 ML models
5. Averages results
6. Returns to frontend

**Why This Works**:
- Hides complexity from users (no need to understand features)
- Keeps model logic consistent (same transformations as training)
- Production-grade: Real ML services use this exact pattern
- Fast: Lookups are O(1), prediction is milliseconds
