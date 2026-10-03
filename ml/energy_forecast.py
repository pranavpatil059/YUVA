"""
VoltSense BEMS - ML energy forecasting prototype
Trains a small Random Forest regressor on synthetic building telemetry.
The dataset is generated for demonstration; replace it with real BEMS data
for production deployment.
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

sample = np.array([[14, 72, 34, 52, 24]])
prediction = model.predict(sample)[0]

print(f"Predicted building load: {prediction:.2f} kW")
print("Model: RandomForestRegressor")
print("Note: demonstration model trained on synthetic telemetry.")
