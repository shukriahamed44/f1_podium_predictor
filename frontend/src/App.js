import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API = 'http://localhost:5000';

function App() {
  const [grid, setGrid] = useState([]);
  const [circuits, setCircuits] = useState([]);
  const [selectedCircuit, setSelectedCircuit] = useState('');
  const [gridPositions, setGridPositions] = useState({});
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load driver grid and circuits on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [gridRes, circuitRes] = await Promise.all([
          axios.get(`${API}/api/grid-info`),
          axios.get(`${API}/api/circuits`)
        ]);
        
        setGrid(gridRes.data);
        setCircuits(circuitRes.data);
        
        // Initialize grid positions object (driver_id -> empty)
        const initialPositions = {};
        gridRes.data.forEach(driver => {
          initialPositions[driver.driver_id] = '';
        });
        setGridPositions(initialPositions);
        
        // Set first circuit as default
        if (circuitRes.data.length > 0) {
          setSelectedCircuit(circuitRes.data[0].id.toString());
        }
      } catch (err) {
        setError('Failed to load data from server. Is the backend running?');
        console.error(err);
      }
    };

    loadData();
  }, []);

  // Handle grid position input change
  const handleGridPositionChange = (driverId, value) => {
    const numValue = parseInt(value);
    const validValue = isNaN(numValue) ? '' : Math.max(1, Math.min(22, numValue));
    setGridPositions(prev => ({
      ...prev,
      [driverId]: validValue
    }));
  };

  // Validate all grid positions are filled and unique
  const validateGridPositions = () => {
    const filled = Object.values(gridPositions).filter(pos => pos !== '');
    
    if (filled.length !== grid.length) {
      setError(`Please enter grid positions for all ${grid.length} drivers`);
      return false;
    }
    
    const positions = filled.map(p => parseInt(p));
    const unique = new Set(positions);
    
    if (unique.size !== positions.length) {
      setError('Each driver must have a unique grid position (1-22)');
      return false;
    }
    
    return true;
  };

  // Handle batch prediction request
  const handlePredictAll = async () => {
    setError(null);
    
    if (!selectedCircuit) {
      setError('Please select a circuit');
      return;
    }
    
    if (!validateGridPositions()) {
      return;
    }

    setLoading(true);
    setResults(null);

    try {
      const response = await axios.post(`${API}/api/predict-batch`, {
        circuit_id: parseInt(selectedCircuit),
        grid_positions: gridPositions
      });

      // Sort results by ensemble probability (highest first)
      const sorted = response.data.predictions.sort((a, b) => b.ensemble_prob - a.ensemble_prob);
      setResults(sorted);
    } catch (err) {
      setError(err.response?.data?.error || 'Batch prediction failed: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏎️ F1 Podium Predictor 2024</h1>
        <p>Predict the podium finishers for all 22 drivers</p>
      </header>

      <div className="container">
        {/* Circuit Selector */}
        <div className="circuit-selector">
          <label htmlFor="circuit">Select Circuit:</label>
          <select
            id="circuit"
            value={selectedCircuit}
            onChange={(e) => setSelectedCircuit(e.target.value)}
            className="circuit-dropdown"
          >
            <option value="">-- Choose Circuit --</option>
            {circuits.map(circ => (
              <option key={circ.id} value={circ.id}>
                {circ.name} ({circ.location})
              </option>
            ))}
          </select>
        </div>

        {/* Grid Positions Input Panel */}
        <div className="grid-input-panel">
          <h2>Enter Grid Positions for All 22 Drivers</h2>
          <p className="instructions">Enter each driver's grid position (1-22), ensuring no duplicates</p>
          
          <div className="grid-inputs">
            {grid.map(driver => (
              <div key={driver.driver_id} className="grid-input-row">
                <label htmlFor={`driver-${driver.driver_id}`}>
                  {driver.driver_name}
                  <span className="constructor"> ({driver.constructor_name})</span>
                </label>
                <input
                  id={`driver-${driver.driver_id}`}
                  type="number"
                  min="1"
                  max="22"
                  value={gridPositions[driver.driver_id] ?? ''}
                  onChange={(e) => handleGridPositionChange(driver.driver_id, e.target.value)}
                  placeholder="1-22"
                  className="grid-input"
                />
              </div>
            ))}
          </div>

          {/* Error Display */}
          {error && <div className="error-box">{error}</div>}

          {/* Predict Button */}
          <button
            onClick={handlePredictAll}
            disabled={loading || grid.length === 0}
            className="predict-btn"
          >
            {loading ? 'Predicting...' : 'PREDICT ALL DRIVERS'}
          </button>
        </div>

        {/* Results Panel */}
        {results && (
          <div className="results-panel">
            <h2>Podium Predictions (Sorted by Winning Chance)</h2>
            
            <table className="results-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Driver</th>
                  <th>Constructor</th>
                  <th>Grid</th>
                  <th>RF %</th>
                  <th>CB %</th>
                  <th>Avg %</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, idx) => (
                  <tr key={result.driver_id} className={result.prediction === 'PODIUM' ? 'podium-row' : ''}>
                    <td className="rank">#{idx + 1}</td>
                    <td className="driver-name">{result.driver_name}</td>
                    <td className="constructor">{result.constructor_name}</td>
                    <td className="grid-pos">P{result.grid}</td>
                    <td className="probability">{result.rf_prob.toFixed(1)}%</td>
                    <td className="probability">{result.cb_prob.toFixed(1)}%</td>
                    <td className="probability ensemble">{result.ensemble_prob.toFixed(1)}%</td>
                    <td className={`verdict ${result.prediction === 'PODIUM' ? 'podium' : 'no-podium'}`}>
                      {result.prediction}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <button onClick={() => setResults(null)} className="reset-btn">
              Try Another Race
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
