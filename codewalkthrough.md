# 📊 F1 PODIUM PREDICTOR - CODE WALKTHROUGH

## Overview
This notebook trains and evaluates machine learning models to predict F1 podium finishes (Top 3). The pipeline includes data loading, feature engineering, model training, hyperparameter tuning, and evaluation using both Random Forest and CatBoost classifiers.

---

## 📝 Section-by-Section Breakdown

### **Section 1: Setup & Data Loading (Cells 1-4)**

#### Cell 1: Install Dependencies
```python
!pip install catboost
```
- Installs CatBoost library needed for gradient boosting model
- Required before importing CatBoost

#### Cell 2: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization

#### Cell 3: Load Dataset
```python
df = pd.read_csv('F1 Races 2020-2024.csv')
df.head()
```
- Loads F1 race data from 2020-2024 (9,839 rows × 34 columns)
- Displays first few rows for inspection
- **Key Variables**: `df` (main dataset)

#### Cell 4: Dataset Shape & Info
```python
print(f"Dataset shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
```
- Shows dataset dimensions
- Lists all feature columns
- **Output**: 9,839 races with 34 features (drivers, constructors, circuits, race conditions, etc.)

---

### **Section 2: Data Exploration (Cells 5-7)**

#### Cell 5: Statistical Summary & Missing Values
```python
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")
```
- Inspect data types and non-null counts
- Identify missing values
- Get min/max/mean/std for numeric columns
- **Key Finding**: No critical missing values; all 9,839 rows have complete core features

#### Cell 6: Class Distribution
```python
podium_counts = df['Top 3 Finish'].value_counts()
class_weights = len(df) / (2 * podium_counts)
```
- Count podium finishes (1) vs non-podium (0)
- Calculate class weights for imbalanced learning
- **Key Finding**: ~20% podium, ~80% non-podium (imbalanced)
- **Solution**: Use `class_weight='balanced'` in models

#### Cell 7: Visualize Class Distribution
```python
df['Top 3 Finish'].value_counts().plot(kind='bar')
plt.title('Podium vs Non-Podium Distribution')
```
- Bar chart of target variable distribution
- Visual confirmation of class imbalance

---

### **Section 3: Feature Engineering & Preprocessing (Cells 8-12)**

#### Cell 8: Remove Data Leakage
```python
post_race_features = [
    'points', 'statusId', 'laps', 'position_previous_race',
    'Weighted_Top_3_Probability', 'Weighted_Top_3_Prob_Length',
    'prom_points_10', 'position', 'raceId', 'points_cons_champ'
]
df = df.drop(columns=post_race_features)
```
- **CRITICAL**: Remove features only available after the race
- These features cause data leakage (model learns post-race info)
- Keep only pre-race features (grid position, driver history, etc.)
- **Impact**: Prevents unrealistic model performance

#### Cell 9: Categorize Features
```python
id_cols = ['driverId', 'constructorId', 'circuitId', 'raceId']
numeric_cols = ['grid', 'driver_age', 'wins', 'points_driver_champ', 
                'Length', 'Turns', 'rainy', ...]
cat_features = ['nationality_encoded', 'nro_cond_escuderia']
```
- ID columns: Used as lookups, not model features
- Numeric columns: Raw numbers (grid position, circuit length, etc.)
- Categorical columns: Encoded values (nationality, drivers per team)
- **Purpose**: Different preprocessing for different feature types

#### Cell 10: Normalize Numeric Features
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_numeric = scaler.fit_transform(X[numeric_cols])
```
- Scale all numeric features to [0, 1] range
- Prevents high-magnitude features from dominating
- **Models affected**: Important for distance-based models

#### Cell 11: One-Hot Encode for Random Forest
```python
X_rf = pd.get_dummies(X, drop_first=True)
```
- Convert categorical features to binary (one-hot encoding)
- Random Forest can't handle categorical data directly
- **Output**: `X_rf` (ready for RF training)

#### Cell 12: Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train_rf, X_test_rf, y_train, y_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)
```
- Split data: 80% training, 20% testing
- `stratify=y`: Maintains class distribution in both sets
- Prevents data leakage (test set completely separate)
- **Output**: Train/test sets with same distribution

---

### **Section 4: Model Training - Random Forest (Cells 13-14)**

#### Cell 13: Build & Train Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=15,          # Limit tree depth
    class_weight='balanced', # Handle class imbalance
    random_state=42
)
rf_model.fit(X_train_rf, y_train)
```
- Creates forest of 200 independent trees
- Each tree votes on prediction → majority vote wins
- `class_weight='balanced'`: Penalizes misclassifying rare class (podium)
- **Model**: `rf_model` (saved for later use)

#### Cell 14: Evaluate Random Forest
```python
y_pred_rf = rf_model.predict(X_test_rf)
y_pred_proba_rf = rf_model.predict_proba(X_test_rf)
print(classification_report(y_test, y_pred_rf))
```
- Predictions on test data
- Probability predictions (0-1 for each class)
- Classification metrics: Accuracy, Precision, Recall, F1-score
- **Output**: Model performance baseline

---

### **Section 5: Feature Importance Analysis (Cells 15-17)**

#### Cell 15: Extract Feature Importances
```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_rf.columns
```
- Get importance score for each feature
- Sort by descending importance
- **Interpretation**: Higher score = more predictive

#### Cell 16: Visualize Top Features
```python
plt.figure(figsize=(10, 6))
plt.bar(range(15), importances[indices[:15]])
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=45)
plt.title('Top 15 Feature Importances')
```
- Bar chart of 15 most important features
- Usually includes: grid position, driver history, constructor stats
- **Key Insight**: Grid position typically most important

#### Cell 17: Feature Importance Statistics
```python
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```
- Print top 10 features with exact importance scores
- Helps understand model decision-making

---

### **Section 6: Model Training - CatBoost (Cells 18-20)**

#### Cell 18: Build & Train CatBoost
```python
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=500,        # 500 boosting rounds
    learning_rate=0.1,
    depth=8,              # Tree depth
    cat_features=cat_features,  # Explicitly specify categorical
    verbose=0
)
cat_model.fit(X_train_cb, y_train)
```
- Gradient boosting (sequential trees, each corrects previous)
- Native categorical feature support (no one-hot needed)
- `iterations=500`: More rounds than RF (learns incrementally)
- **Model**: `cat_model` (alternative to RF)

#### Cell 19: Evaluate CatBoost
```python
y_pred_cat = cat_model.predict(X_test_cb)
y_pred_proba_cat = cat_model.predict_proba(X_test_cb)
print(classification_report(y_test, y_pred_cat))
```
- Test set predictions
- Compare performance to Random Forest
- Usually CatBoost performs better on this problem

#### Cell 20: CatBoost Feature Importance
```python
cat_importances = cat_model.feature_importances_
plt.bar(range(15), cat_importances[:15])
```
- Feature importances from CatBoost
- May differ from RF (different algorithm = different priorities)

---

### **Section 7: Hyperparameter Tuning (Cells 21-22)**

#### Cell 21: Define Hyperparameter Grid
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}
```
- Test different combinations of hyperparameters
- n_estimators: Number of trees
- max_depth: Tree depth limit
- min_samples_split: Minimum samples to split a node

#### Cell 22: GridSearchCV (⚠️ Resource Intensive)
```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_rf, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```
- Tests all param combinations (3×3×3 = 27 models)
- 5-fold cross-validation for each
- **Output**: Best hyperparameters found
- ⚠️ Takes time (can be hours for large datasets)

---

### **Section 8: Advanced Modeling (Cells 23-25)**

#### Cell 23: Ensemble Stacking
```python
# Combine predictions from RF and CatBoost
ensemble_proba = (y_pred_proba_rf + y_pred_proba_cat) / 2
ensemble_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
```
- Average predictions from both models
- Often beats individual models (combines strengths)
- **Approach**: Voting/averaging ensembles

#### Cell 24: Model Comparison
```python
models = {
    'Random Forest': (y_pred_rf, y_pred_proba_rf),
    'CatBoost': (y_pred_cat, y_pred_proba_cat),
    'Ensemble': (ensemble_pred, ensemble_proba)
}
# Compare ROC curves, metrics for each
```
- Side-by-side performance comparison
- ROC curves for visual comparison
- **Finding**: Usually Ensemble ≥ CatBoost > Random Forest

#### Cell 25: Model Improvements & Recommendations
```python
# Summary of findings:
# - Data leakage removed: ✓
# - Class imbalance handled: ✓
# - Feature engineering complete: ✓
# - Hyperparameter tuned: ✓
# - Ready for production: ✓
```
- Summarize strengths/weaknesses
- Recommendations for deployment
- Next steps for production

---

### **Section 9: Model Saving & Production (Cells 26-28)**

#### Cell 26: Save Trained Models
```python
import joblib
joblib.dump(rf_model, 'backend/randomforest_f1_model.pkl')
cat_model.save_model('backend/catboost_f1_model.cbm', format='cbm')
joblib.dump(driver_lookup, 'backend/driver_lookup.pkl')
joblib.dump(constructor_lookup, 'backend/constructor_lookup.pkl')
joblib.dump(circuit_lookup, 'backend/circuit_lookup.pkl')
```
- Serialize trained models to disk
- Lookup tables for id→name mapping
- **Files Created**:
  - `randomforest_f1_model.pkl` (RF model)
  - `catboost_f1_model.cbm` (CatBoost model)
  - `driver_lookup.pkl` (driver statistics)
  - `constructor_lookup.pkl` (constructor statistics)
  - `circuit_lookup.pkl` (circuit statistics)

#### Cell 27: Load Models & Define Prediction Function
```python
# Load models
cat_model = CatBoostClassifier()
cat_model.load_model('backend/catboost_f1_model.cbm')
driver_lookup = joblib.load('backend/driver_lookup.pkl')

def predict_podium(driver_id, circuit_id, grid_position):
    """Predict podium chance for single driver"""
    # Get circuit stats, driver stats
    # Build feature vector with ONLY pre-race columns
    # Get predictions from both models
    # Average and return result
    return {"driver_id": driver_id, "ensemble_prob": 0.70, ...}

# Example
result = predict_podium(driver_id=1, circuit_id=1, grid_position=1)
```
- Load serialized models from disk
- Define `predict_podium()` function:
  - Takes driver ID, circuit, grid position
  - Returns prob from RF, CB, and ensemble
  - **Key**: Uses SAME feature engineering as training
- Single driver prediction example

#### Cell 28: Batch Predictions (All 22 Drivers)
```python
def predict_all_drivers(circuit_id, grid_positions_dict):
    """Predict for all 22 drivers at once"""
    results = []
    for driver_id, grid_pos in grid_positions_dict.items():
        pred = predict_podium(driver_id, circuit_id, grid_pos)
        results.append(pred)
    
    df = pd.DataFrame(results)
    df = df.sort_values('ensemble_prob', ascending=False)
    return df

# Test: Abu Dhabi with drivers in grid positions 1-22
grid_positions = {1: 1, 4: 2, 807: 3, ..., 907: 22}
results = predict_all_drivers(circuit_id=1, grid_positions_dict=grid_positions)

# Show top podium candidates
print(results[results['ensemble_prob'] > 0.50])
```
- Batch prediction for all 22 drivers
- Sorts by winning probability (highest first)
- Identifies PODIUM candidates (>50% probability)
- **Output**: DataFrame with all 22 drivers ranked

---

## 🎯 Key Takeaways

| Aspect | Value |
|--------|-------|
| **Dataset Size** | 9,839 races × 24 features (after removing leakage) |
| **Target Variable** | Top 3 Finish (binary: 1=podium, 0=non-podium) |
| **Class Distribution** | ~20% podium, ~80% non-podium (imbalanced) |
| **Best Model** | CatBoost (usually highest ROC-AUC) |
| **Key Features** | Grid position, driver experience, constructor performance |
| **Data Leakage** | ✅ Removed post-race features |
| **Feature Engineering** | ✅ Normalization, encoding, history aggregation |
| **Production Ready** | ✅ Models serialized, prediction functions defined |

---

## 🚀 Usage

### **In Notebook**
```python
# Single driver prediction
result = predict_podium(driver_id=1, circuit_id=1, grid_position=1)
# Output: {"driver_id": 1, "ensemble_prob": 0.75, "prediction": "PODIUM"}

# All 22 drivers (batch)
grid_positions = {1: 1, 4: 2, 807: 3, ..., 907: 22}
results_df = predict_all_drivers(circuit_id=1, grid_positions_dict=grid_positions)
# Shows all 22 drivers ranked by podium probability
```

### **Via Flask API**
```bash
# Start backend
cd backend && python app.py

# POST /api/predict-batch
# Body: {"circuit_id": 1, "grid_positions": {1: 1, 4: 2, ...}}
# Response: All 22 drivers with predictions
```

### **Via React Web UI**
1. Run `npm start` in frontend/
2. Select circuit from dropdown
3. Enter grid positions for all 22 drivers
4. Click "PREDICT ALL DRIVERS"
5. View results table sorted by podium chance

---

## 📂 File Structure
```
F1 ML/
├── GP26_ML_Project.ipynb              # Main notebook (28 cells)
├── codewalkthrough.md                 # This file
├── F1 Races 2020-2024.csv             # Training data (9,839 races)
├── backend/
│   ├── app.py                         # Flask API server
│   ├── mappings.py                    # 22 drivers + 11 constructors + 24 circuits
│   ├── randomforest_f1_model.pkl      # Trained RF model
│   ├── catboost_f1_model.cbm          # Trained CatBoost model
│   ├── driver_lookup.pkl              # Driver stats (age, wins, TOP3%, etc.)
│   ├── constructor_lookup.pkl         # Constructor stats
│   └── circuit_lookup.pkl             # Circuit stats (length, turns)
└── frontend/
    ├── src/
    │   ├── App.js                     # React app (22 driver grid inputs)
    │   └── App.css                    # Styling
    └── package.json
```

---

## 🔄 Data Pipeline Flow

```
Raw CSV
  ↓
[Load & Explore]
  ↓
[Remove Data Leakage] → Drop post-race features
  ↓
[Feature Engineering] → Normalize, encode, aggregate history
  ↓
[Split Train/Test] → 80-20 stratified split
  ↓
[Train Models] → Random Forest + CatBoost
  ↓
[Evaluate & Tune] → GridSearchCV, compare metrics
  ↓
[Save Models] → Pickle (.pkl) and CatBoost (.cbm) formats
  ↓
[Make Predictions] → predict_podium() & predict_all_drivers()
  ↓
[Deploy] → Flask backend + React frontend
```

---

## ⚠️ Important Notes

1. **Data Leakage**: Post-race features (position, points, statusId) removed
   - These only exist AFTER the race
   - Would give model unfair advantage (cheating)
   
2. **Class Imbalance**: Only ~20% podium finishes
   - Handled with `class_weight='balanced'`
   - Accuracy alone not reliable (always ~80% by guessing non-podium)
   - Use ROC-AUC, F1-score, precision/recall instead
   
3. **Feature Consistency**: 
   - Training features != Prediction features
   - Predictions use SAME columns & preprocessing as training
   - Mismatch would cause errors
   
4. **Lookup Tables**: 
   - Store driver/constructor/circuit stats separately
   - Avoid re-computing during prediction
   - Enable quick batch predictions

---

## 📊 Model Architecture

### **Random Forest**
- Algorithm: Ensemble of 200 independent decision trees
- Pros: Fast, interpretable feature importance, handles non-linearity
- Cons: Can overfit on noise

### **CatBoost**
- Algorithm: Gradient boosting (sequential trees, cumulative corrections)
- Pros: Often better performance, native categorical support
- Cons: Slower training, less interpretable

### **Ensemble**
- Algorithm: Average predictions from both models
- Pros: Generally beats individual models
- Cons: Slower (need both models)
