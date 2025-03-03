// React App (src/App.js)
import React, { useState } from 'react';
import './App.css';

function App() {
  const [features, setFeatures] = useState([0, 0, 0, 0, 0, 0, 0, 0]);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value);
    setFeatures(newFeatures);
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    }
  };

  return (
    <div className="App">
      <h1>Model Prediction</h1>
      <div className="features-input">
        {features.map((feature, index) => (
          <div key={index}>
            <label>Feature {index + 1}:</label>
            <input
              type="number"
              value={feature}
              onChange={(e) => handleInputChange(index, e.target.value)}
            />
          </div>
        ))}
      </div>
      <button onClick={handleSubmit}>Predict</button>
      {error && <div className="error">{error}</div>}
      {prediction !== null && (
        <div className="prediction">Prediction: {prediction}</div>
      )}
    </div>
  );
}

export default App;