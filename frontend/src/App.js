import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import './App.css';

const API = 'http://localhost:5000';

// 3-letter driver codes
const DRIVER_ABBR = {
  1: 'HAM', 4: 'ALO', 807: 'HUL', 815: 'PER', 822: 'BOT',
  830: 'VER', 832: 'SAI', 839: 'OCO', 840: 'STR', 842: 'GAS',
  844: 'LEC', 846: 'NOR', 847: 'RUS', 848: 'ALB',
  900: 'COL', 901: 'BOR', 902: 'BEA', 903: 'PIA',
  904: 'ANT', 905: 'LAW', 906: 'LIN', 907: 'HAD',
};

const CONSTRUCTOR_ORDER = [
  'McLaren', 'Ferrari', 'Red Bull', 'Mercedes',
  'Aston Martin', 'Alpine', 'Williams', 'Racing Bulls',
  'Haas', 'Audi', 'Cadillac',
];

const CONSTRUCTOR_COLORS = {
  'McLaren':      '#FF8000',
  'Ferrari':      '#DC0000',
  'Red Bull':     '#3671C6',
  'Mercedes':     '#00D5B8',
  'Aston Martin': '#358C75',
  'Alpine':       '#FF87BC',
  'Williams':     '#64C4FF',
  'Racing Bulls': '#6692FF',
  'Haas':         '#A8A8A8',
  'Audi':         '#C0C0C0',
  'Cadillac':     '#C00024',
};

// Teams that need dark text on their color bg
const LIGHT_TEAM_BG = new Set(['Mercedes', 'Alpine', 'Williams', 'Haas', 'Audi']);

const CIRCUIT_FLAGS = {
  1: '🇦🇺', 3: '🇧🇭', 4: '🇪🇸', 6: '🇲🇨', 7: '🇨🇦',
  9: '🇬🇧', 11: '🇭🇺', 13: '🇧🇪', 14: '🇮🇹', 15: '🇸🇬',
  17: '🇨🇳', 18: '🇧🇷', 21: '🇮🇹', 22: '🇯🇵', 24: '🇦🇪',
  32: '🇲🇽', 39: '🇳🇱', 69: '🇺🇸', 70: '🇦🇹', 73: '🇦🇿',
  77: '🇸🇦', 78: '🇶🇦', 79: '🇺🇸',
};

// ──────────────────────────────────────────────────────────────
// GridBox — one position slot with inline custom dropdown
// ──────────────────────────────────────────────────────────────
function GridBox({
  position, driverObj, sortedDriversList, assignedDriverIds,
  onAssign, isOpen, onOpen, onClose,
}) {
  const wrapRef = useRef(null);
  const abbr      = driverObj ? (DRIVER_ABBR[driverObj.driver_id] || '???') : null;
  const teamColor = driverObj ? (CONSTRUCTOR_COLORS[driverObj.constructor_name] || '#555') : null;
  const isPole    = position === 1;
  const textDark  = driverObj && LIGHT_TEAM_BG.has(driverObj.constructor_name);

  const handleToggle = (e) => {
    e.stopPropagation();
    isOpen ? onClose() : onOpen(position);
  };

  const handleSelect = (driverId) => {
    onAssign(position, String(driverId));
    onClose();
  };

  const handleClear = (e) => {
    e.stopPropagation();
    onAssign(position, '');
    onClose();
  };

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const h = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) onClose();
    };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [isOpen, onClose]);

  return (
    <div className="gbox-wrap" ref={wrapRef}>
      <button
        className={[
          'gbox',
          driverObj ? 'gbox--filled' : '',
          isPole    ? 'gbox--pole'   : '',
          isOpen    ? 'gbox--open'   : '',
        ].join(' ')}
        style={driverObj ? {
          background: teamColor,
          borderColor: teamColor,
          color: textDark ? '#111' : '#fff',
        } : undefined}
        onClick={handleToggle}
        aria-label={`Position ${position}`}
      >
        {driverObj
          ? <span className="gbox__abbr">{abbr}</span>
          : <span className="gbox__num">{String(position).padStart(2, '0')}</span>
        }
        {isOpen && !driverObj && <span className="gbox__caret">▾</span>}
      </button>

      {isOpen && (
        <div className="gdrop" onClick={e => e.stopPropagation()}>
          {CONSTRUCTOR_ORDER.map(constructor => {
            const teamDrivers = sortedDriversList.filter(d => d.constructor_name === constructor);
            if (!teamDrivers.length) return null;
            const color = CONSTRUCTOR_COLORS[constructor] || '#555';

            return teamDrivers.map(d => {
              const isCurrent = driverObj?.driver_id === d.driver_id;
              const taken     = assignedDriverIds.has(d.driver_id) && !isCurrent;
              const lastName  = d.driver_name.split(' ').slice(1).join(' ').toUpperCase();

              return (
                <button
                  key={d.driver_id}
                  className={['gdrop__item', taken ? 'gdrop__item--taken' : '', isCurrent ? 'gdrop__item--current' : ''].join(' ')}
                  onClick={() => !taken && (isCurrent ? handleClear({ stopPropagation: () => {} }) : handleSelect(d.driver_id))}
                  disabled={taken}
                >
                  <span className="gdrop__bar" style={{ background: color }} />
                  <span className="gdrop__abbr">{DRIVER_ABBR[d.driver_id]}</span>
                  <span className="gdrop__name">{lastName}</span>
                  {taken    && <span className="gdrop__tag gdrop__tag--taken">P{Object.entries({}).find(([,v]) => Number(v) === d.driver_id)?.[0] || '✓'}</span>}
                  {isCurrent && <span className="gdrop__tag gdrop__tag--cur">✕</span>}
                </button>
              );
            });
          })}
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// App
// ──────────────────────────────────────────────────────────────
export default function App() {
  const [allDrivers,      setAllDrivers]      = useState([]);
  const [circuits,        setCircuits]        = useState([]);
  const [selectedCircuit, setSelectedCircuit] = useState('');
  const [gridPositions,   setGridPositions]   = useState(() => {
    const o = {}; for (let i = 1; i <= 22; i++) o[i] = ''; return o;
  });
  const [results,     setResults]     = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [openDropdown, setOpenDropdown] = useState(null); // null | 'circuit' | 1-22

  const circuitBtnRef = useRef(null);

  // Sorted driver list: by constructor order
  const sortedDriversList = useMemo(() => {
    if (!allDrivers.length) return [];
    return [...allDrivers].sort((a, b) => {
      const ai = CONSTRUCTOR_ORDER.indexOf(a.constructor_name);
      const bi = CONSTRUCTOR_ORDER.indexOf(b.constructor_name);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
        || a.driver_name.localeCompare(b.driver_name);
    });
  }, [allDrivers]);

  // Which driver IDs are already assigned
  const assignedDriverIds = useMemo(() =>
    new Set(Object.values(gridPositions).filter(v => v !== '').map(Number))
  , [gridPositions]);

  // Load data
  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/grid-info`),
      axios.get(`${API}/api/circuits`),
    ]).then(([g, c]) => {
      setAllDrivers(g.data);
      setCircuits(c.data);
      if (c.data.length) setSelectedCircuit(String(c.data[0].id));
    }).catch(() => setError('Cannot reach backend. Is the Flask server running?'));
  }, []);

  // Close circuit dropdown on outside click
  useEffect(() => {
    if (openDropdown !== 'circuit') return;
    const h = (e) => {
      if (circuitBtnRef.current && !circuitBtnRef.current.contains(e.target))
        setOpenDropdown(null);
    };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [openDropdown]);

  const handleAssign = useCallback((slot, driverId) => {
    setGridPositions(prev => {
      const next = { ...prev };
      if (driverId !== '') {
        Object.keys(next).forEach(s => {
          if (next[s] === driverId && Number(s) !== slot) next[s] = '';
        });
      }
      next[slot] = driverId;
      return next;
    });
  }, []);

  const filledCount = useMemo(
    () => Object.values(gridPositions).filter(v => v !== '').length,
    [gridPositions]
  );

  const handlePredict = async () => {
    setError(null);
    if (!selectedCircuit)   { setError('Select a circuit first.'); return; }
    if (filledCount < 22)   { setError(`Fill all 22 positions (${filledCount}/22).`); return; }
    setLoading(true);
    setResults(null);

    const payload = {};
    Object.entries(gridPositions).forEach(([slot, did]) => {
      if (did !== '') payload[did] = parseInt(slot);
    });

    try {
      const res = await axios.post(`${API}/api/predict-batch`, {
        circuit_id: parseInt(selectedCircuit),
        grid_positions: payload,
      });
      setResults(res.data.predictions.sort((a, b) => b.ensemble_prob - a.ensemble_prob));
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setGridPositions(() => { const o = {}; for (let i = 1; i <= 22; i++) o[i] = ''; return o; });
    setError(null);
  };

  const circuitObj = circuits.find(c => String(c.id) === selectedCircuit);

  // Row layout matching F1 grid: even positions top, odd positions bottom
  // Both rows go from high (back) to low (front) → P22…P02 and P21…P01
  const evenRow = Array.from({ length: 11 }, (_, i) => 22 - i * 2); // 22,20…2
  const oddRow  = Array.from({ length: 11 }, (_, i) => 21 - i * 2); // 21,19…1

  return (
    <div className="app" onClick={() => setOpenDropdown(null)}>

      {/* ══════════ HEADER ══════════ */}
      <header className="header" onClick={e => e.stopPropagation()}>
        <div className="header__inner">

          {/* F1 Wordmark */}
          <div className="f1-wordmark">
            <svg className="f1-svg" viewBox="0 0 60 30" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Stylized F1 */}
              <text x="0" y="26" fontFamily="'Space Grotesk', sans-serif" fontWeight="900" fontSize="30" fill="white" fontStyle="italic">F1</text>
            </svg>
            <span className="f1-sub">PODIUM PREDICTOR</span>
          </div>

          {/* Circuit selector */}
          <div className="circuit-wrap" ref={circuitBtnRef}>
            <button
              className={`circuit-pill ${openDropdown === 'circuit' ? 'circuit-pill--open' : ''}`}
              onClick={() => setOpenDropdown(openDropdown === 'circuit' ? null : 'circuit')}
            >
              {circuitObj ? (
                <>
                  <span className="cp-flag">{CIRCUIT_FLAGS[circuitObj.id] || '🏁'}</span>
                  <span className="cp-name">{circuitObj.name.toUpperCase()}</span>
                </>
              ) : (
                <span className="cp-name">SELECT CIRCUIT</span>
              )}
              <svg className={`cp-arrow ${openDropdown === 'circuit' ? 'cp-arrow--up' : ''}`} viewBox="0 0 16 16" fill="currentColor">
                <path d="M3 5l5 5 5-5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round"/>
              </svg>
            </button>

            {openDropdown === 'circuit' && (
              <div className="circuit-drop" onClick={e => e.stopPropagation()}>
                {circuits.map(c => (
                  <button
                    key={c.id}
                    className={`cdrop__item ${String(c.id) === selectedCircuit ? 'cdrop__item--active' : ''}`}
                    onClick={() => { setSelectedCircuit(String(c.id)); setOpenDropdown(null); }}
                  >
                    <span className="cdi-flag">{CIRCUIT_FLAGS[c.id] || '🏁'}</span>
                    <span className="cdi-name">{c.name}</span>
                    <span className="cdi-loc">{c.location}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ══════════ MAIN ══════════ */}
      <main className="main" onClick={e => e.stopPropagation()}>

        {!results ? (
          <div className="grid-card">

            {/* Two-row starting grid */}
            <div className="grid-track">

              {/* Row 1 — Even positions (P22→P02) */}
              <div className="grid-row grid-row--even">
                {evenRow.map(pos => {
                  const did = gridPositions[pos];
                  const dObj = did ? allDrivers.find(d => d.driver_id === Number(did)) : null;
                  return (
                    <GridBox
                      key={pos}
                      position={pos}
                      driverObj={dObj}
                      sortedDriversList={sortedDriversList}
                      assignedDriverIds={assignedDriverIds}
                      onAssign={handleAssign}
                      isOpen={openDropdown === pos}
                      onOpen={setOpenDropdown}
                      onClose={() => setOpenDropdown(null)}
                    />
                  );
                })}
              </div>

              {/* Track centre line */}
              <div className="track-line" />

              {/* Row 2 — Odd positions (P21→P01), staggered right */}
              <div className="grid-row grid-row--odd">
                {oddRow.map(pos => {
                  const did = gridPositions[pos];
                  const dObj = did ? allDrivers.find(d => d.driver_id === Number(did)) : null;
                  return (
                    <GridBox
                      key={pos}
                      position={pos}
                      driverObj={dObj}
                      sortedDriversList={sortedDriversList}
                      assignedDriverIds={assignedDriverIds}
                      onAssign={handleAssign}
                      isOpen={openDropdown === pos}
                      onOpen={setOpenDropdown}
                      onClose={() => setOpenDropdown(null)}
                    />
                  );
                })}
              </div>
            </div>

            {/* Progress bar + CTA */}
            <div className="grid-footer">
              <div className="progress-bar-wrap">
                <div className="progress-bar" style={{ width: `${(filledCount / 22) * 100}%` }} />
              </div>
              <div className="progress-label">
                <span className={filledCount === 22 ? 'prog-done' : ''}>{filledCount}</span>
                <span className="prog-sep">/22 drivers placed</span>
              </div>
              {error && <div className="error-msg">{error}</div>}
              <button
                className="predict-btn"
                onClick={handlePredict}
                disabled={loading || filledCount < 22}
              >
                {loading ? (
                  <><span className="spinner" />RUNNING MODELS…</>
                ) : 'PREDICT PODIUM'}
              </button>
            </div>
          </div>
        ) : (

          /* ══════════ RESULTS ══════════ */
          <div className="results-card">
            <div className="results-header">
              <h2 className="results-title">PODIUM PREDICTIONS</h2>
              <span className="results-circuit">{circuitObj?.name || ''}</span>
            </div>

            <div className="results-table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>DRIVER</th>
                    <th>TEAM</th>
                    <th>GRID</th>
                    <th>RF</th>
                    <th>CB</th>
                    <th>ENSEMBLE</th>
                    <th>VERDICT</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => {
                    const tc = CONSTRUCTOR_COLORS[r.constructor_name] || '#555';
                    return (
                      <tr key={r.driver_id} className={r.prediction === 'PODIUM' ? 'tr--podium' : ''}>
                        <td className="td-rank">
                          {i < 3
                            ? <span className={`medal medal--${i}`}>{i + 1}</span>
                            : `#${i + 1}`}
                        </td>
                        <td className="td-driver">
                          <span className="td-abbr" style={{ color: tc }}>
                            {DRIVER_ABBR[r.driver_id] || '???'}
                          </span>
                          {r.driver_name}
                        </td>
                        <td className="td-team">
                          <span className="team-pip" style={{ background: tc }} />
                          {r.constructor_name}
                        </td>
                        <td className="td-grid">P{r.grid}</td>
                        <td className="td-prob">{r.rf_prob.toFixed(1)}%</td>
                        <td className="td-prob">{r.cb_prob.toFixed(1)}%</td>
                        <td className="td-ens">{r.ensemble_prob.toFixed(1)}%</td>
                        <td>
                          <span className={`verdict ${r.prediction === 'PODIUM' ? 'verdict--yes' : 'verdict--no'}`}>
                            {r.prediction === 'PODIUM' ? '🏆 PODIUM' : 'NO PODIUM'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <button className="reset-btn" onClick={handleReset}>← RESET GRID</button>
          </div>
        )}
      </main>
    </div>
  );
}
