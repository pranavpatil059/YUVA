from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
CORS(app)

rng = np.random.default_rng(42)
n = 1200
hour = rng.integers(0, 24, n)
occupancy = rng.uniform(0, 100, n)
temperature = rng.uniform(18, 42, n)
humidity = rng.uniform(25, 85, n)
setpoint = rng.uniform(22, 26, n)

hvac_kw = 8 + 0.16 * occupancy + 1.8 * np.maximum(temperature - setpoint, 0)
hvac_kw += rng.normal(0, 2.0, n)
load_kw = 12 + 0.08 * occupancy + 0.35 * np.maximum(temperature - 24, 0)
load_kw += 2.5 * np.sin((hour - 7) * np.pi / 12) + rng.normal(0, 1.2, n)

X = np.column_stack([hour, occupancy, temperature, humidity, setpoint])
y = np.maximum(load_kw + 0.55 * hvac_kw, 0)

model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
model.fit(X, y)

@app.get("/")
def health():
    return jsonify({"service": "VoltSense ML Forecast API", "model": "RandomForestRegressor", "status": "online"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        values = [
            float(data.get("hour", 14)),
            float(data.get("occupancy", 72)),
            float(data.get("temperature", 34)),
            float(data.get("humidity", 52)),
            float(data.get("setpoint", 24)),
        ]
    except (TypeError, ValueError):
        return jsonify({"error": "Inputs must be numeric"}), 400

    prediction = float(model.predict(np.array([values]))[0])
    return jsonify({
        "predicted_load_kw": round(prediction, 2),
        "model": "RandomForestRegressor",
        "inputs": dict(zip(["hour","occupancy","temperature","humidity","setpoint"], values)),
        "training_data": "synthetic BEMS telemetry for hackathon prototype"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
