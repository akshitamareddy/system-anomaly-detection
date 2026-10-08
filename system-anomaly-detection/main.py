"""Isolation Forest anomaly detection on synthetic server telemetry."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 1000
metrics = pd.DataFrame({
    "cpu_percent": np.clip(rng.normal(42, 12, n), 0, 100),
    "memory_percent": np.clip(rng.normal(58, 10, n), 0, 100),
    "network_mbps": np.clip(rng.lognormal(2.6, .55, n), 0, None),
})
spikes = rng.choice(n, 35, replace=False)
metrics.loc[spikes, "cpu_percent"] = rng.uniform(90, 100, len(spikes))
metrics.loc[spikes, "memory_percent"] = rng.uniform(88, 100, len(spikes))
metrics.loc[spikes, "network_mbps"] *= 6
model = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
metrics["anomaly"] = model.fit_predict(metrics) == -1
metrics["anomaly_score"] = -model.decision_function(metrics[["cpu_percent", "memory_percent", "network_mbps"]])
metrics.to_csv(OUTPUT / "server_anomalies.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(metrics.index, metrics.cpu_percent, s=10, alpha=.5, label="Normal/observed CPU")
flagged = metrics[metrics.anomaly]
ax.scatter(flagged.index, flagged.cpu_percent, color="red", s=24, label="Flagged anomalies")
ax.set(xlabel="Observation", ylabel="CPU utilization (%)", title="Server anomaly detection")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "anomalies.png", dpi=150)
print(f"Flagged {len(flagged)} of {len(metrics)} observations. Results saved to outputs/.")
