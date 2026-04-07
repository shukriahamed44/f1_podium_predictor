# F1 ML Project: Code Walkthrough

This notebook trains two different ML algorithms to predict whether an F1 driver will finish in the top 3 (podium). It's a binary classification problem using race data from 2020-2024.

---

## 1. Installation & Imports

### Cell 1: Install CatBoost
```python
!pip install catboost
```
**What it does**: CatBoost isn't pre-installed in Google Colab, so this downloads and installs it. The exclamation mark tells Jupyter to run this as a shell command rather than Python.

---

### Cell 2: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
```

**Line by line**:
- **pandas**: Data manipulation (it's the Python equivalent of Excel on steroids)
- **numpy**: Numerical operations and arrays
- **matplotlib.pyplot**: Plotting graphs
- **train_test_split**: Splits data into training (80%) and testing (20%) sets
- **accuracy_score**: Calculates how many predictions were correct
- **classification_report**: Shows precision, recall, and F1-score per class
- **confusion_matrix**: Shows how many predictions were right vs wrong
- **ConfusionMatrixDisplay & RocCurveDisplay**: Visualization helpers
- **RandomForestClassifier**: The first algorithm (ensemble of decision trees)
- **CatBoostClassifier**: The second algorithm (gradient boosting optimized for categorical data)

---

## 2. Data Loading

### Cell 3: Load Dataset
```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Machine_Learning/F1 Races 2020-2024.csv', encoding='latin-1')
df.head()
```

**What's happening**:
- **drive.mount()**: Connects Google Colab to your Google Drive account so you can access files stored there
- **pd.read_csv()**: Reads the CSV file (comma-separated values) into a DataFrame (a table-like object pandas uses)
- **encoding='latin-1'**: Tells pandas how to decode special characters in the file
- **df.head()**: Prints the first 5 rows so you can eyeball the data

---

### Cell 4: Data Structure Check
```python
df.info()
df.shape
```

**What it does**:
- **df.info()**: Shows column names, data types (int, float, string), and null values. Useful for spotting issues early
- **df.shape**: Returns (rows, columns) — tells you the dataset size. You probably have ~1000+ race records

---

## 3. Data Preparation

### Cell 5: Split Target & Features + Lookup Table Strategy
```python
y = df["Top 3 Finish"]
X = df.drop(columns=["Top 3 Finish"])

# KEEP IDs for lookup table (will use later for user predictions)
keep_ids = ['driverId', 'constructorId', 'circuitId']

# REMOVE POST-RACE FEATURES (Data Leakage Prevention)
post_race_features = [...]

# REMOVE TEMPORAL FEATURES ONLY
temporal_features = ['year', 'date', 'round', 'nationality_encoded', 'raceId']

# Store reference data before dropping IDs from model features
reference_data = df[keep_ids + ['driver_age', 'wins', ...]].drop_duplicates()
```

**Key Strategy**:
- **Keep driverId, constructorId, circuitId** NOT as model features, but as **lookup keys**
- These IDs create the **lookup table** that bridges user input to model features
- **Lookup table workflow**: User enters driver → lookup system finds that driver's ID → fetches their stats → feeds to model

**Why this setup?**
- Users enter simple inputs: "Driver name", "Constructor name", "Circuit name", grid position
- System looks up the ID from those names
- Uses ID to fetch all derived features (age, wins, podium %, track length, etc.)
- Feeds only the actual predictive features to the model

This is the **embedding strategy**: Hide complexity from users, expose simplicity.

---

### Cell 6: IMPROVEMENT C - Better Missing Value Handling
```python
# For time-series data (driver/constructor performance), use forward fill then backward fill
X_sorted = X.sort_values(['driverId', 'year', 'round'], errors='ignore').reset_index(drop=True)
X_sorted = X_sorted.fillna(method='ffill')  # Forward fill for driver performance trends
X_sorted = X_sorted.fillna(method='bfill')  # Backward fill for remaining NaNs

# For remaining NaNs, fill with mean (numeric columns only)
X = X_sorted.fillna(X_sorted.mean(numeric_only=True))

# Remove outliers: drivers in their first race (zero historical performance)
initial_rows = len(X)
X = X[(X['Driver Top 3 Finish Percentage (Last Year)'] > 0) | 
      (X['Driver Top 3 Finish Percentage (This Year till last race)'] > 0) |
      (X['wins'] > 0)].copy()
removed_rows = initial_rows - len(X)

if removed_rows > 0:
    y = y.loc[X.index]  # Align y with cleaned X

print("Data Quality Improvements:")
print(f"  ✓ Used forward/backward fill for time-series data")
print(f"  ✓ Filled remaining NaNs with column means")
print(f"  ✓ Removed {removed_rows} outlier rows (first-time drivers with no history)")
print(f"  ✓ Final dataset shape: {X.shape}")
```

**Why this matters**:
- **Simple mean imputation** (the old approach) treats all missing values the same way
- **Forward/backward fill**: For time-series data (performance metrics across the season), we can assume a driver's performance stays somewhat consistent until it changes. So if January's data is missing, use February's data from before (forward fill) or after (backward fill)
- **Removing first-time drivers**: A driver in their first race has 0% of anything (0 wins, 0 podiums). The model can't learn from data that's all zeros, so it's better to exclude them
- This approach is more sophisticated than basic mean imputation and respects the temporal nature of F1 data

---

### Cell 7: Visualize Class Balance
```python
y.value_counts().plot(kind="bar")
plt.title("Class Distribution: Podium vs Non-Podium")
plt.xlabel("Top 3 Finish")
plt.ylabel("Count")
plt.show()
```

**Line by line**:
- **y.value_counts()**: Counts how many 1s and 0s you have in the target (e.g., 300 podiums, 700 non-podiums)
- **.plot(kind="bar")**: Creates a bar chart
- **plt.title/xlabel/ylabel**: Labels for the chart
- **plt.show()**: Displays the chart

Why? Imbalanced data (way more 0s than 1s) can confuse models. You want to know if this is an issue.

---

### Cell 8: IMPROVEMENT A - Feature Engineering

```python
# Create interaction features
X['grid_x_driver_consistency'] = X['grid'] * X['Driver Top 3 Finish Percentage (This Year till last race)']
X['grid_x_constructor_performance'] = X['grid'] * X['Constructor Top 3 Finish Percentage (This Year till last race)']
X['age_x_wins'] = X['driver_age'] * X['wins']
X['track_length_x_wins'] = X['Length'] * X['wins']

# Normalize performance metrics (0-1 scale)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

performance_cols = [
    'Driver Top 3 Finish Percentage (Last Year)',
    'Constructor Top 3 Finish Percentage (Last Year)',
    'Driver Top 3 Finish Percentage (This Year till last race)',
    'Constructor Top 3 Finish Percentage (This Year till last race)',
    'grid',
    'driver_age'
]

X[performance_cols] = scaler.fit_transform(X[performance_cols])
```

**What it does**:
- **Interaction features**: Multiplying two features together can reveal hidden relationships
  - `grid_x_driver_consistency`: Starting position × how often a driver gets podiums. A good driver starting from pole position is more likely to finish top 3
  - `age_x_wins`: Older, experienced drivers with wins are more dangerous
  - These don't just add information—they encode relationships
- **Normalization**: Scales features to 0-1 range
  - Without scaling, high numbers (like driver age: 35) dominate over small decimals (like percentages: 0.45)
  - Normalized data helps the model treat all metrics fairly

**Why it matters**: 
- Raw features are often insufficient; combined features reveal what experienced analysts see
- Normalization prevents large-scale features from drowning out small ones

---

## 4. Algorithm 1: Random Forest

### Cell 9: One-Hot Encoding
```python
X_rf = pd.get_dummies(X, drop_first=True)
```

**What it does**:
Random Forest can't understand text (like "Wet", "Dry" for weather). This converts categorical columns into numbers:
- If a column has "Wet, Dry, Overcast" → creates 3 new YES/NO columns
- **drop_first=True**: Drops one column to avoid redundancy (if it's not "Dry" and not "Overcast", it must be "Wet")

Example transformation:
```
Weather       →  Weather_Dry  Weather_Overcast
Wet                 0               0
Dry                 1               0
Overcast            0               1
```

---

### Cell 9: Split Data
```python
X_train_rf, X_test_rf, y_train, y_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)
```

**Line by line**:
- **test_size=0.2**: 20% goes to testing, 80% to training (standard practice)
- **random_state=42**: Makes the split reproducible (same split every time). The number 42 is arbitrary
- **stratify=y**: Ensures both train and test sets have the same proportion of podiums/non-podiums (if 30% are podiums overall, both sets get 30%)
- **Returns 4 things**: Training features, test features, training target, test target

---

### Cell 10: Train Random Forest
```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train_rf, y_train)
```

**What it does**:
- **n_estimators=200**: Creates 200 individual decision trees. Each tree votes, and the majority vote wins
- **max_depth=15**: Each tree can be at most 15 levels deep (prevents overfitting by stopping the tree from getting too complex)
- **random_state=42**: Again, for reproducibility
- **.fit()**: Actually trains the model using training data

Why Random Forest? It's robust, handles many features well, and doesn't require much tuning.

---

### Cell 11: Evaluate Random Forest
```python
y_pred_rf = rf_model.predict(X_test_rf)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```

**What it does**:
- **rf_model.predict()**: Uses the trained model to make predictions on test data it's never seen
- **accuracy_score()**: Percentage of correct predictions (if accuracy = 0.85, it got 85% right)
- **classification_report()**: Shows detailed stats:
  - **Precision**: Of the podiums it predicted, how many were actually podiums?
  - **Recall**: Of the actual podiums in the test set, how many did it find?
  - **F1-score**: Harmonic mean of precision and recall (single number that balances both)

---

### Cell 12: Confusion Matrix
```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Confusion Matrix – Random Forest")
plt.show()
```

**What it shows** (a 2x2 grid):
```
           Predicted 0  Predicted 1
Actual 0      TN          FP
Actual 1      FN          TP
```
- **TN (True Negative)**: Correctly predicted "no podium"
- **FP (False Positive)**: Wrongly predicted "podium" (missed out)
- **FN (False Negative)**: Wrongly predicted "no podium" (should've caught it)
- **TP (True Positive)**: Correctly predicted "podium"

---

### Cell 13: Feature Importance
```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X_rf.columns[indices])
plt.title("Top 10 Feature Importance – Random Forest")
plt.show()
```

**Line by line**:
- **rf_model.feature_importances_**: Gets a score for each feature (0-1) indicating how much it influenced predictions
- **np.argsort()**: Sorts features by importance score
- **[-10:]**: Takes the last 10 (the top 10 most important)
- **plt.barh()**: Horizontal bar chart (easier to read long feature names)
- **plt.yticks()**: Labels the bars with feature names

Real-world insight: If "Starting Grid Position" has high importance, your model thinks where you start matters more than engine power. That's useful information for understanding the sport.

---

## 5. Algorithm 2: CatBoost

### Cell 14: Identify Categorical Features
```python
cat_features = [
    i for i, col in enumerate(X.columns)
    if X[col].dtype == "object"
]
```

**What it does**:
CatBoost is special—it handles categorical data (text) natively, no encoding needed. This identifies which columns are categorical:
- **enumerate(X.columns)**: Gets both the index number and column name of each column
- **dtype == "object"**: Text columns have dtype "object" in pandas
- **List comprehension**: Creates a list of indices where columns are text

Output example: `[3, 5, 7]` means columns 3, 5, and 7 are categorical.

---

### Cell 15: Split Data (CatBoost)
```python
X_train_cb, X_test_cb, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**What it does**:
Same as Random Forest, but this time using the original `X` (with text columns still intact, not one-hot encoded). CatBoost will handle the text automatically.

Note: `y_train` and `y_test` are reused—they're the same split targets from the RF section. Python just overwrites the variable names.

---

### Cell 16: Train CatBoost
```python
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=8,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=100
)

cat_model.fit(
    X_train_cb, y_train,
    cat_features=cat_features
)
```

**Line by line**:
- **iterations=500**: CatBoost trains iteratively (like 500 rounds of refinement). Higher = potentially better but slower
- **learning_rate=0.1**: How much it adjusts after each iteration. Lower = slower learning, less likely to overshoot
- **depth=8**: Maximum tree depth (CatBoost also uses trees)
- **loss_function="Logloss"**: Binary classification metric (minimizes log loss)
- **eval_metric="Accuracy"**: Metric to track during training
- **verbose=100**: Prints progress every 100 iterations (so you don't think it's frozen)
- **cat_features=**: Tells CatBoost "these columns are categorical, handle them yourself"

Why CatBoost over one-hot encoding? It handles categorical data more intelligently, often producing better results.

---

### Cell 17: Evaluate CatBoost
```python
y_pred_cat = cat_model.predict(X_test_cb)

print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))
```

**What it does**:
Identical logic to Random Forest—predicts on unseen test data and prints accuracy + detailed metrics. You can now compare: which model won?

---

### Cell 18: Confusion Matrix (CatBoost)
```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cat)
plt.title("Confusion Matrix – CatBoost")
plt.show()
```

Same visualization as Random Forest. Visual comparison can reveal which model makes different types of errors.

---

## 7. User Prediction Pipeline (Lookup + Embed)

### Cell 21: Create Lookup Tables
```python
driver_lookup = df[['driverId', 'driver_age', 'wins', ...]].drop_duplicates()
constructor_lookup = df[['constructorId', ...]].drop_duplicates()
circuit_lookup = df[['circuitId', 'Turns', 'Length', ...]].drop_duplicates()

joblib.dump(driver_lookup, "driver_lookup.pkl")
joblib.dump(constructor_lookup, "constructor_lookup.pkl")
joblib.dump(circuit_lookup, "circuit_lookup.pkl")
```

**What it does**:
The lookup tables are **bridges between user input and model features**.

**User Input Flow**:
```
User: "Max Verstappen, Red Bull, Monaco, Grid 3"
         ↓
Lookup: driverId=1 → fetch age, wins, podium %
        constructorId=3 → fetch constructor performance
        circuitId=6 → fetch track length, turns
         ↓
Embed: Combine grid position + fetched stats into feature vector
         ↓
Predict: Feed to both models → output probabilities
```

**Why lookup tables?**
- Users don't know IDs—they know names
- Lookup table converts names → IDs → all features automatically
- Model never sees raw IDs; only derived features (which actually predict)

### Cell 22: Prediction Function
```python
def predict_podium_finish(driver_id, constructor_id, circuit_id, grid_position):
    # Step 1: Lookup driver stats by ID
    driver_stats = driver_lookup[driver_lookup['driverId'] == driver_id]
    
    # Step 2: Lookup constructor stats by ID
    constructor_stats = constructor_lookup[constructor_lookup['constructorId'] == constructor_id]
    
    # Step 3: Lookup circuit stats by ID
    circuit_stats = circuit_lookup[circuit_lookup['circuitId'] == circuit_id]
    
    # Step 4: Embed - combine all features
    input_features = pd.DataFrame({...all fetched stats...})
    
    # Step 5: Predict
    rf_prob = rf_model.predict_proba(input_features)
    cat_prob = cat_model.predict_proba(input_features)
    ensemble = (rf_prob + cat_prob) / 2
    
    return {'podium_chance': f"{ensemble[1]:.1%}", 'prediction': 'PODIUM/NO PODIUM'}
```

**Flow in Plain English**:
1. Accept 4 simple inputs: driver ID, constructor ID, circuit ID, grid position
2. Use IDs to look up all the driver/constructor/circuit stats from saved tables
3. Combine those stats with grid position
4. Feed the combined features to both trained models
5. Average their predictions for final result
6. Return probability as percentage to user

---

## 8. Model Persistence

### Cell 19: Save Models
```python
cat_model.save_model("catboost_f1_model.cbm")

import joblib
joblib.dump(rf_model, "randomforest_f1_model.pkl")

joblib.dump(X.columns.tolist(), "features.pkl")
```

**Why save models?**
Training takes time. Once trained, save it so you can reuse it later without retraining.

**Line by line**:
- **cat_model.save_model()**: CatBoost's native save method (saves as `.cbm`)
- **joblib.dump()**: Generic Python serialization tool (saves as `.pkl`)
  - Saves the Random Forest model
  - Saves the original feature column names (you'll need these when making new predictions—the order matters)

---

### Cell 20: Download from Colab
```python
from google.colab import files

files.download("features.pkl")
files.download("randomforest_f1_model.pkl")
files.download("catboost_f1_model.cbm")
```

**What it does**:
Google Colab runs in the cloud. This downloads your trained models to your local computer so you can use them later (e.g., in a Flask app predicting real race outcomes).

---

### Cell 21: Empty Cell
No code—just there if you want to add something later.

---

## Summary: What This Notebook Does

1. **Loads F1 race data** from 2020-2024
2. **Cleans it** (fills missing values)
3. **Visualizes class balance** (how many podiums vs non-podiums?)
4. **Trains Random Forest** (200 trees, handles text via one-hot encoding)
5. **Trains CatBoost** (500 iterations, handles text natively)
6. **Evaluates both** with accuracy, precision, recall, F1-score, and confusion matrices
7. **Compares feature importance** (which race stats matter most?)
8. **Saves both models** for later use

The goal: Build a predictor that, given race conditions and driver stats, can forecast whether a driver finishes in the top 3.

---

## Key Takeaways

- **Random Forest** is straightforward but requires manual encoding of categorical data
- **CatBoost** is more sophisticated and handles text columns automatically
- **Always evaluate on test data**, never train/test on the same set (it's cheating)
- **Monitor multiple metrics** (accuracy alone can be misleading with imbalanced data)
- **Save your models** so you don't have to retrain them every time you want a prediction
