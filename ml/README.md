# VoltSense BEMS — ML Forecast Prototype

This folder contains the machine-learning prototype behind the **Energy Forecasting** layer shown in the VoltSense BEMS dashboard.

### Model
- **Random Forest Regressor**
- Inputs: hour, occupancy, outdoor temperature, humidity and HVAC setpoint
- Output: predicted building electrical load in kW
- Training data: synthetic BEMS telemetry generated for the hackathon demonstration

### Run
```bash
pip install -r requirements.txt
python energy_forecast.py
```

The current static Render dashboard uses simulated telemetry for its live UI. This Python model is the repository-side ML prototype and can be connected to real BACnet/Modbus/MQTT data in a backend deployment.
