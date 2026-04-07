"""
F1 ML Prediction Backend - 22 Driver Grid Predictions (2026)
Fixed feature alignment for proper model predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from mappings import DRIVERS_2024, CONSTRUCTORS_2024, CIRCUITS_2024, get_2024_grid
from catboost import CatBoostClassifier

app = Flask(__name__)
CORS(app)

print("🔄 Loading models and lookup tables...")

# Load Random Forest
try:
    rf_model = joblib.load('randomforest_f1_model.pkl')
    print("✓ Random Forest loaded")
except Exception as e:
    print(f"❌ RF Error: {e}")
    rf_model = None

# Load CatBoost
try:
    cat_model = CatBoostClassifier()
    cat_model.load_model('catboost_f1_model.cbm', format='cbm')
    print("✓ CatBoost loaded")
except Exception as e:
    print(f"❌ CatBoost Error: {e}")
    cat_model = None

# Load lookup tables
try:
    driver_lookup = joblib.load('driver_lookup.pkl')
    constructor_lookup = joblib.load('constructor_lookup.pkl')
    circuit_lookup = joblib.load('circuit_lookup.pkl')
    print("✓ Lookup tables loaded")
except Exception as e:
    print(f"❌ Lookup Error: {e}")
    driver_lookup = None
    constructor_lookup = None
    circuit_lookup = None

print("✓ Backend ready\n")

# ============================================================================
# PREDICT FUNCTION - Reconstructs full training schema features
# ============================================================================

def predict_driver(driver_id, circuit_id, grid_pos):
    """Predict podium chance for single driver"""
    
    if rf_model is None or cat_model is None:
        return None
    
    # Get circuit stats
    circuit_data = circuit_lookup[circuit_lookup['circuitId'] == circuit_id]
    if circuit_data.empty:
        return None
    
    track_length = float(circuit_data['Length'].values[0])
    track_turns = int(circuit_data['Turns'].values[0])
    
    # Get driver info
    if driver_id not in DRIVERS_2024:
        return None
    
    driver_info = DRIVERS_2024[driver_id]
    constructor_id = driver_info["constructor_id"]
    
    # Lookup driver stats
    driver_data = driver_lookup[driver_lookup['driverId'] == driver_id]
    
    if driver_data.empty:
        # New driver - use defaults based on position in grid
        driver_age = 24 + (grid_pos // 5)  # Estimate age based on grid pos
        wins = 0
        point_champ = 0
        top3_last = 5.0 + grid_pos  # Rough estimate
        top3_this = 5.0 + grid_pos
        pos_last = 15.0
        pos_this = float(grid_pos)
        const_top3_last = 10.0
        const_top3_this = 10.0
        const_pos_last = 8.0
        const_pos_this = 8.0
    else:
        row = driver_data.iloc[-1]
        driver_age = float(row['driver_age'])
        wins = float(row['wins'])
        point_champ = float(row['points_driver_champ'] if 'points_driver_champ' in row.index else 0)
        top3_last = float(row['Driver Top 3 Finish Percentage (Last Year)'] if 'Driver Top 3 Finish Percentage (Last Year)' in row.index else 10.0)
        top3_this = float(row['Driver Top 3 Finish Percentage (This Year till last race)'] if 'Driver Top 3 Finish Percentage (This Year till last race)' in row.index else 10.0)
        pos_last = float(grid_pos) if 'Driver Avg position (Last Year)' not in row.index else float(row['Driver Avg position (Last Year)'])
        pos_this = float(grid_pos) if 'Driver Average Position (This Year till last race)' not in row.index else float(row['Driver Average Position (This Year till last race)'])
        
        # Constructor stats
        if constructor_id:
            const_data = constructor_lookup[constructor_lookup['constructorId'] == constructor_id]
            if not const_data.empty:
                crow = const_data.iloc[-1]
                const_top3_last = float(crow['Constructor Top 3 Finish Percentage (Last Year)'] if 'Constructor Top 3 Finish Percentage (Last Year)' in crow.index else 10.0)
                const_top3_this = float(crow['Constructor Top 3 Finish Percentage (This Year till last race)'] if 'Constructor Top 3 Finish Percentage (This Year till last race)' in crow.index else 10.0)
                const_pos_last = float(crow.get('Constructor Avg position (Last Year)', 8.0) if 'Constructor Avg position (Last Year)' in crow.index else 8.0)
                const_pos_this = float(crow.get('Constructor Average Position (This Year till last race)', 8.0) if 'Constructor Average Position (This Year till last race)' in crow.index else 8.0)
            else:
                const_top3_last = 10.0
                const_top3_this = 10.0
                const_pos_last = 8.0
                const_pos_this = 8.0
        else:
            const_top3_last = 10.0
            const_top3_this = 10.0
            const_pos_last = 8.0
            const_pos_this = 8.0
    
    # Build feature vector with ALL columns the models expect
    # This reconstructs the full training schema
    today = datetime(2026, 4, 8)
    
    base_features = {
        'year': 2026,
        'round': 5,
        'circuitId': circuit_id,
        'date': '2026-04-08',
        'rainy': 0,
        'Turns': track_turns,
        'Length': track_length,
        'driverId': driver_id,
        'constructorId': int(constructor_id) if constructor_id else 999,
        'grid': float(grid_pos),
        'Driver Top 3 Finish Percentage (Last Year)': top3_last,
        'Constructor Top 3 Finish Percentage (Last Year)': const_top3_last,
        'Driver Top 3 Finish Percentage (This Year till last race)': top3_this,
        'Constructor Top 3 Finish Percentage (This Year till last race)': const_top3_this,
        'driver_age': driver_age,
        'nationality_encoded': 0,
        'wins_cons': 0,
        'points_cons_champ': 0,
        'wins': wins,
        'points_driver_champ': point_champ,
        'laps': 0,
        'statusId': 1,
        'position_previous_race': int(grid_pos),
        'nro_cond_escuderia': 0,
        'raceId': circuit_id,
        'points': 0,
        'prom_points_10': 0,
        'Driver Avg position (Last Year)': pos_last,
        'Constructor Avg position (Last Year)': const_pos_last,
        'Driver Average Position (This Year till last race)': pos_this,
        'Constructor Average Position (This Year till last race)': const_pos_this,
        'Weighted_Top_3_Probability': (50.0 - grid_pos * 2) / 100.0,  # Estimate based on grid
        'Weighted_Top_3_Prob_Length': (50.0 - grid_pos * 2) / 100.0,
    }
    
    features = pd.DataFrame([base_features])
    
    print(f"DEBUG - Driver {driver_id}: age={driver_age}, wins={wins}, grid={grid_pos}, top3_last={top3_last:.1f}%")
    
    # Random Forest prediction
    try:
        features_rf = pd.get_dummies(features, drop_first=True)
        
        # Add missing columns more efficiently (avoid dataframe fragmentation)
        missing_cols = {col: 0 for col in rf_model.feature_names_in_ if col not in features_rf.columns}
        if missing_cols:
            features_rf = pd.concat([features_rf, pd.DataFrame({k: [v] for k, v in missing_cols.items()})], axis=1)
        
        # Reorder columns to match training
        features_rf = features_rf[rf_model.feature_names_in_]
        
        rf_proba = rf_model.predict_proba(features_rf)[0]
        rf_podium = float(rf_proba[1]) if len(rf_proba) > 1 else 0.0
        print(f"  ✓ RF: {rf_podium*100:.1f}%")
    except Exception as rf_err:
        print(f"  ❌ RF: {str(rf_err)[:50]}")
        rf_podium = 0.0
    
    # CatBoost prediction
    try:
        cat_proba = cat_model.predict_proba(features)[0]
        cat_podium = float(cat_proba[1]) if len(cat_proba) > 1 else 0.0
        print(f"  ✓ CB: {cat_podium*100:.1f}%")
    except Exception as cat_err:
        print(f"  ❌ CB: {str(cat_err)[:50]}")
        cat_podium = 0.0
    
    ensemble = (rf_podium + cat_podium) / 2
    
    return {
        "driver_id": driver_id,
        "driver_name": driver_info["name"],
        "constructor_name": CONSTRUCTORS_2024.get(constructor_id, "Unknown"),
        "grid": grid_pos,
        "rf_prob": round(rf_podium * 100, 1),
        "cb_prob": round(cat_podium * 100, 1),
        "ensemble_prob": round(ensemble * 100, 1),
        "value": ensemble,
        "prediction": "PODIUM" if ensemble > 0.5 else "NO PODIUM"
    }

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/api/grid-info', methods=['GET'])
def grid_info():
    """Get 2026 driver grid"""
    return jsonify(get_2024_grid())

@app.route('/api/circuits', methods=['GET'])
def circuits():
    """Get circuits"""
    data = [{"id": cid, "name": v["name"], "location": v["location"]} for cid, v in CIRCUITS_2024.items()]
    return jsonify(sorted(data, key=lambda x: x["name"]))

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict podium for all drivers. Input: {circuit_id, grid_positions: {driver_id: grid}}"""
    try:
        data = request.json
        circuit_id = int(data['circuit_id'])
        grid_pos = {int(k): int(v) for k, v in data.get('grid_positions', {}).items()}
        
        results = []
        for driver_id, grid in grid_pos.items():
            try:
                pred = predict_driver(driver_id, circuit_id, grid)
                if pred:
                    results.append(pred)
            except Exception as driver_err:
                print(f"Error predicting driver {driver_id}: {driver_err}")
                continue
        
        results.sort(key=lambda x: x["value"], reverse=True)
        for r in results:
            del r["value"]
        
        return jsonify({
            "circuit_id": circuit_id,
            "circuit_name": CIRCUITS_2024.get(circuit_id, {}).get("name", "Unknown"),
            "predictions": results,
            "total": len(results)
        })
    
    except Exception as e:
        print(f"Batch error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "rf": rf_model is not None,
        "cb": cat_model is not None,
        "lookups": driver_lookup is not None
    })

if __name__ == '__main__':
    print("="*70)
    print("F1 PREDICTOR - 22 DRIVERS (2026)")
    print("="*70)
    print("Server: http://localhost:5000")
    print("  POST /api/predict-batch")
    print("  GET  /api/grid-info")
    print("  GET  /api/circuits")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
