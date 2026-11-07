# Parking Availability Prediction

End-to-end ML app that predicts car-park availability from **real sensor occupancy labels**
(UCI Parking Birmingham), with a Streamlit dashboard and a Melbourne live-bay map for context.

> This project deliberately does **not** invent labels. Availability is derived from measured
> `Occupancy / Capacity`. Evaluation uses a chronological hold-out so metrics reflect
> genuine predictive skill, not shuffled noise.

## What you can honestly claim

- Built a full ML application: data pipeline → feature engineering → multiple models → evaluation → interactive dashboard.
- Trained classifiers and regressors on public Birmingham NCP occupancy data (~35k readings, 30 car parks).
- Used time-aware validation and avoided label leakage (no current occupancy in features).
- Compared Logistic Regression, Random Forest, Histogram Gradient Boosting, and Linear SVM on hold-out ROC-AUC / F1; regressed occupancy rate with Ridge / RF / HGB.

## Datasets

| Dataset | Role | Source |
|--------|------|--------|
| **Parking Birmingham** (UCI) | Training & evaluation | [archive.ics.uci.edu/dataset/482](https://archive.ics.uci.edu/dataset/482/parking+birmingham) |
| **Melbourne bay sensors** | Live map context (not train labels) | [data.melbourne.vic.gov.au](https://data.melbourne.vic.gov.au/explore/dataset/on-street-parking-bay-sensors/) |

**Label:** `available = 1` if `occupancy / capacity < 0.85`, else `0` (nearly full).

**Features:** hour / day / weekend (with cyclical encodings), car-park identity, capacity, and **lagged** occupancy from prior readings only.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Data (if not already under data/raw/)
python scripts/download_data.py

# Features → models → metrics
python src/data_prep.py
python src/train.py

# Dashboard
streamlit run app/streamlit_app.py
```

## Project layout

```
app/streamlit_app.py      # Interactive dashboard
src/config.py             # Paths & constants
src/data_prep.py          # Cleaning + feature engineering
src/train.py              # Chronological train/eval
scripts/download_data.py  # Fetch public CSVs
data/raw/                 # Birmingham + Melbourne downloads
models/                   # Fitted pipelines (.joblib)
reports/metrics.json      # Hold-out metrics
```

## Integrity notes

- Do **not** add synthetic `np.random` labels.
- Do **not** put passwords or admin credentials in this README or the repo.
- Prefer chronological splits for time-series parking data; random K-fold overstates performance.
- Short-horizon AUC can look very high because occupancy is persistent — always compare against the **persistence (lag-1)** and **calendar-only** baselines in `reports/metrics.json`.

## Example hold-out results

After `python src/train.py`, open `reports/metrics.json`. Typical pattern on Birmingham:

| Setup | What it means | Rough ROC-AUC |
|------|----------------|---------------|
| Persistence lag-1 | "Same as last reading" | strong baseline |
| Calendar + zone only | Time-of-day demand patterns | strong, no recent occupancy |
| Calendar + lags | Short-horizon forecast | best (beats persistence slightly) |

Exact numbers depend on your run; quote the JSON, not memory.

## License / attribution

- Birmingham data: UK Open Government Licence / UCI CC BY 4.0 distribution.
- Melbourne sensors: City of Melbourne, Creative Commons Attribution.
