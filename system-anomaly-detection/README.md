# System Anomaly Detection

An unsupervised machine-learning demonstration that flags unusual combinations of CPU utilization, memory usage, and network throughput using **Isolation Forest**.

## Features
- Generates reproducible synthetic server telemetry with injected spikes.
- Fits an Isolation Forest and exports anomaly flags and scores.
- Saves a visualization of flagged observations.

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```

Outputs: `outputs/server_anomalies.csv` and `outputs/anomalies.png`.

## Method and limitations
The example uses **synthetic** metrics rather than production monitoring data. Anomaly scores indicate unusual observations, **not confirmed crashes**. Real deployment requires timestamped monitoring data, temporal validation, alert thresholds, and evaluation against labeled incidents.
