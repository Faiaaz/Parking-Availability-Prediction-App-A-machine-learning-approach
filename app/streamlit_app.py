from pathlib import Path
import sys

import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    FEATURES_PATH,
    MELBOURNE_CSV,
    METRICS_PATH,
    MODELS_DIR,
    OCCUPANCY_FULL_THRESHOLD,
    REPORTS_DIR,
)
from data_prep import FEATURE_COLUMNS, load_birmingham, build_features  # noqa: E402

st.set_page_config(
    page_title="Parking Availability — Real Occupancy Models",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Visual system ---------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --ink: #0f1c18;
  --moss: #1f6f54;
  --moss-deep: #14503c;
  --sand: #e8f0ec;
  --amber: #c47a2c;
  --slate: #3d524a;
  --card: #f7faf8;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #d7ebe2 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #f3e6d4 0%, transparent 50%),
    linear-gradient(180deg, #f4f7f5 0%, #eef3f0 100%);
}

h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.02em; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

.hero {
  padding: 1.4rem 1.6rem 1.2rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #12382c 0%, #1f6f54 55%, #2a8a68 100%);
  color: #f4fff9;
  margin-bottom: 1.2rem;
  box-shadow: 0 18px 40px rgba(18, 56, 44, 0.22);
}
.hero h1 {
  color: #f4fff9 !important;
  font-size: 2.1rem;
  margin: 0 0 0.35rem 0;
}
.hero p { margin: 0; opacity: 0.92; max-width: 52rem; }
.badge {
  display: inline-block;
  margin-top: 0.75rem;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 0.5rem 0 1.25rem;
}
.metric-pill {
  background: var(--card);
  border: 1px solid #d5e4dc;
  border-radius: 14px;
  padding: 0.9rem 1rem;
}
.metric-pill .label { font-size: 0.75rem; color: var(--slate); text-transform: uppercase; letter-spacing: 0.06em; }
.metric-pill .value { font-family: 'Fraunces', serif; font-size: 1.55rem; color: var(--moss-deep); margin-top: 0.15rem; }
.footnote { color: var(--slate); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_features():
    if FEATURES_PATH.exists():
        try:
            return pd.read_parquet(FEATURES_PATH)
        except Exception:
            pass
    csv_path = FEATURES_PATH.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date_time"])
    return build_features()


@st.cache_data
def load_melbourne():
    if not MELBOURNE_CSV.exists():
        return None
    df = pd.read_csv(MELBOURNE_CSV, sep=";")
    # location is "lat, lon"
    coords = df["location"].astype(str).str.split(",", expand=True)
    df["lat"] = pd.to_numeric(coords[0], errors="coerce")
    df["lon"] = pd.to_numeric(coords[1], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    status = df["status_description"].fillna("Unknown").str.strip()
    df["status_clean"] = status.replace({"": "Unknown"})
    df["is_available"] = df["status_clean"].str.lower().eq("unoccupied")
    return df


@st.cache_resource
def load_bundle():
    path = MODELS_DIR / "best_bundle.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except AttributeError as exc:
        st.error(
            "Saved models were pickled with a different scikit-learn version. "
            "Re-run `python src/train.py`, then click **Clear cache** in the "
            "app menu (⋮) and reload."
        )
        st.caption(f"Load error: {exc}")
        return None


def roc_figure(metrics: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance (AUC=0.5)",
            line=dict(dash="dash", color="#9aa8a1"),
        )
    )
    for name, result in metrics["classification"].items():
        curves_path = REPORTS_DIR / f"{name}_curves.joblib"
        if not curves_path.exists():
            continue
        curves = joblib.load(curves_path)
        fpr, tpr, _ = roc_curve(curves["y_true"], curves["y_proba"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={result['roc_auc']:.3f})",
            )
        )
    fig.update_layout(
        title="ROC curves on chronological hold-out",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def occupancy_heatmap(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["hour"] = tmp["date_time"].dt.hour
    tmp["dow"] = tmp["date_time"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        tmp.pivot_table(index="dow", columns="hour", values="occupancy_rate", aggfunc="mean")
        .reindex(order)
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="YlGnBu",
        aspect="auto",
        labels=dict(color="Occupancy rate"),
        title="Mean occupancy by weekday × hour (Birmingham)",
    )
    fig.update_layout(height=380, template="plotly_white")
    return fig


# --- Sidebar ---------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore data", "Model lab", "Predict", "Melbourne live map"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
**Label (real)**  
`available = 1` if occupancy / capacity **&lt; {OCCUPANCY_FULL_THRESHOLD:.0%}**

**Split**  
Chronological hold-out (last 20% of time)

**Source**  
[UCI Parking Birmingham](https://archive.ics.uci.edu/dataset/482/parking+birmingham)
"""
)

metrics = load_metrics()
features = load_features()
bundle = load_bundle()

# --- Pages -----------------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
<div class="hero">
  <div class="badge">Real labels · Chronological evaluation</div>
  <h1>Parking Availability Prediction</h1>
  <p>
    Models trained on NCP car-park occupancy in Birmingham (UCI), not coin-flip labels.
    Predict whether a car park still has space — and how full it will be — from time,
    zone, and recent occupancy history.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning("Models not trained yet. Run `python src/data_prep.py` then `python src/train.py`.")
        st.stop()

    best = metrics["best_classifier"]
    best_m = metrics["classification"][best]
    best_r = metrics["regression"][metrics["best_regressor"]]
    cal_key = metrics.get("best_calendar_only_classifier")
    cal_m = metrics["classification"].get(cal_key, {}) if cal_key else {}
    persist = metrics.get("baselines", {}).get("classification", {})

    st.markdown(
        f"""
<div class="metric-strip">
  <div class="metric-pill"><div class="label">Best short-horizon AUC</div><div class="value">{best_m['roc_auc']:.3f}</div></div>
  <div class="metric-pill"><div class="label">Calendar-only AUC</div><div class="value">{cal_m.get('roc_auc', float('nan')):.3f}</div></div>
  <div class="metric-pill"><div class="label">Persistence AUC</div><div class="value">{persist.get('roc_auc', float('nan')):.3f}</div></div>
  <div class="metric-pill"><div class="label">Best R² (occupancy)</div><div class="value">{best_r['r2']:.3f}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Best model: `{best}` · Calendar-only: `{cal_key}` · "
        "Persistence = predict from the previous reading only."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", f"{metrics['n_train']:,}")
    c2.metric("Test rows", f"{metrics['n_test']:,}")
    c3.metric("Car parks", f"{features['parking_zone'].nunique()}")

    st.markdown(
        f"""
<span class="footnote">
Train window: {metrics['train_start'][:10]} → {metrics['train_end'][:10]} ·
Test window: {metrics['test_start'][:10]} → {metrics['test_end'][:10]} ·
{metrics['label_definition']}
</span>
""",
        unsafe_allow_html=True,
    )

    st.plotly_chart(roc_figure(metrics), use_container_width=True)

    st.subheader("Why this is honest research engineering")
    st.markdown(
        """
- **Labels come from sensors**: `Occupancy` and `Capacity` are in the public dataset — no `np.random` labels.
- **No label leakage**: models never see *current* occupancy; only calendar features, zone, and *prior* readings.
- **Time-aware split**: the test set is the most recent 20% of timestamps, not a random shuffle.
- **Baselines reported**: persistence (lag-1) and calendar-only models, so high short-horizon AUC is not mistaken for magic.
- **Two tasks**: classify nearly-full vs available, and regress occupancy rate for finer comparison.
"""
    )

elif page == "Explore data":
    st.header("Birmingham occupancy patterns")
    st.caption("UCI Parking Birmingham — 30 NCP car parks, Oct–Dec 2016, ~half-hourly readings.")

    raw = load_birmingham()
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(occupancy_heatmap(raw), use_container_width=True)
    with right:
        daily = (
            raw.set_index("date_time")
            .groupby("parking_zone")["occupancy_rate"]
            .resample("D")
            .mean()
            .reset_index()
        )
        top_zones = raw.groupby("parking_zone")["occupancy_rate"].mean().nlargest(6).index
        fig = px.line(
            daily[daily["parking_zone"].isin(top_zones)],
            x="date_time",
            y="occupancy_rate",
            color="parking_zone",
            title="Daily mean occupancy (busiest zones)",
        )
        fig.update_layout(height=380, template="plotly_white", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    zone = st.selectbox("Inspect a car park", sorted(raw["parking_zone"].unique()))
    zone_df = raw[raw["parking_zone"] == zone].copy()
    zone_df["available"] = zone_df["occupancy_rate"] < OCCUPANCY_FULL_THRESHOLD
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=zone_df["date_time"],
            y=zone_df["occupancy_rate"],
            mode="lines",
            name="Occupancy rate",
            line=dict(color="#1f6f54", width=1.4),
        )
    )
    fig.add_hline(
        y=OCCUPANCY_FULL_THRESHOLD,
        line_dash="dash",
        line_color="#c47a2c",
        annotation_text=f"Full threshold ({OCCUPANCY_FULL_THRESHOLD:.0%})",
    )
    fig.update_layout(
        title=f"{zone} — occupancy over time",
        yaxis_title="Occupancy / capacity",
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    avail = (zone_df["occupancy_rate"] < OCCUPANCY_FULL_THRESHOLD).mean()
    st.info(
        f"**{zone}**: capacity {int(zone_df['capacity'].iloc[0])}, "
        f"mean occupancy {zone_df['occupancy_rate'].mean():.1%}, "
        f"available {avail:.1%} of readings."
    )

elif page == "Model lab":
    st.header("Model comparison")
    if metrics is None:
        st.warning("Train models first.")
        st.stop()

    cls_rows = []
    for name, m in metrics["classification"].items():
        cls_rows.append(
            {
                "model": name,
                "feature_set": m.get("feature_set", "calendar+lags"),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "avg_precision": m["average_precision"],
            }
        )
    if "baselines" in metrics and "classification" in metrics["baselines"]:
        b = metrics["baselines"]["classification"]
        cls_rows.append(
            {
                "model": "persistence_lag1",
                "feature_set": "baseline",
                "accuracy": b["accuracy"],
                "precision": b["precision"],
                "recall": b["recall"],
                "f1": b["f1"],
                "roc_auc": b["roc_auc"],
                "avg_precision": b["average_precision"],
            }
        )
    cls_df = pd.DataFrame(cls_rows).sort_values("roc_auc", ascending=False)
    st.subheader("Classification — will space still be available?")
    st.dataframe(
        cls_df.style.format(
            {
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "roc_auc": "{:.3f}",
                "avg_precision": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig = px.bar(
        cls_df.melt(
            id_vars=["model", "feature_set"],
            value_vars=["roc_auc", "f1", "accuracy"],
        ),
        x="model",
        y="value",
        color="variable",
        barmode="group",
        title="Classifier metrics on hold-out (includes baselines)",
        color_discrete_sequence=["#1f6f54", "#c47a2c", "#3d6ea8"],
    )
    fig.update_layout(template="plotly_white", yaxis_range=[0, 1], height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model
    best = metrics["best_classifier"]
    cm = np.array(metrics["classification"][best]["confusion_matrix"])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Full", "Available"],
        y=["Full", "Available"],
        title=f"Confusion matrix — {best}",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    reg_rows = []
    for name, m in metrics["regression"].items():
        reg_rows.append(
            {
                "model": name,
                "feature_set": m.get("feature_set", "calendar+lags"),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            }
        )
    if "baselines" in metrics and "regression" in metrics["baselines"]:
        b = metrics["baselines"]["regression"]
        reg_rows.append(
            {
                "model": "persistence_lag1",
                "feature_set": "baseline",
                "mae": b["mae"],
                "rmse": b["rmse"],
                "r2": b["r2"],
            }
        )
    reg_df = pd.DataFrame(reg_rows).sort_values("r2", ascending=False)
    st.subheader("Regression — predicted occupancy rate")
    st.dataframe(
        reg_df.style.format({"mae": "{:.4f}", "rmse": "{:.4f}", "r2": "{:.3f}"}),
        use_container_width=True,
    )

    # Pred vs actual for best regressor
    reg_name = metrics["best_regressor"]
    curves = joblib.load(REPORTS_DIR / f"{reg_name}_curves.joblib")
    sample_idx = np.random.RandomState(0).choice(len(curves["y_true"]), size=min(2500, len(curves["y_true"])), replace=False)
    scatter = pd.DataFrame(
        {
            "actual": np.array(curves["y_true"])[sample_idx],
            "predicted": np.array(curves["y_pred"])[sample_idx],
        }
    )
    sfig = px.scatter(
        scatter,
        x="actual",
        y="predicted",
        opacity=0.35,
        title=f"Predicted vs actual occupancy — {reg_name}",
    )
    sfig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(color="#c47a2c", dash="dash"))
    )
    sfig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(sfig, use_container_width=True)

elif page == "Predict":
    st.header("What-if predictor")
    if bundle is None:
        st.warning("Train models first (`python src/train.py`).")
        st.stop()

    zones = sorted(features["parking_zone"].unique())
    zone = st.selectbox("Car park", zones)
    zone_hist = features[features["parking_zone"] == zone].sort_values("date_time")
    capacity = int(zone_hist["capacity"].iloc[-1])

    c1, c2, c3 = st.columns(3)
    hour = c1.slider("Hour of day", 0, 23, 12)
    dow = c2.selectbox(
        "Day of week",
        list(range(7)),
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
        index=2,
    )
    month = c3.selectbox("Month", [10, 11, 12], index=1)

    # Default lags from that zone's median history at similar hours
    similar = zone_hist[zone_hist["hour"] == hour]
    default_lag = float(similar["occupancy_rate"].median()) if len(similar) else float(zone_hist["occupancy_rate"].median())
    lag1 = st.slider("Recent occupancy rate (lag 1)", 0.0, 1.0, float(np.clip(default_lag, 0, 1)), 0.01)
    lag2 = st.slider("Prior occupancy rate (lag 2)", 0.0, 1.0, float(np.clip(default_lag * 0.95, 0, 1)), 0.01)
    roll3 = (lag1 + lag2 + default_lag) / 3

    row = pd.DataFrame(
        [
            {
                "capacity": capacity,
                "hour": hour,
                "dayofweek": dow,
                "month": month,
                "is_weekend": int(dow >= 5),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
                "time_of_day": float(hour),
                "occ_rate_lag1": lag1,
                "occ_rate_lag2": lag2,
                "occ_rate_roll3": roll3,
                "occ_rate_delta_lag": lag1 - lag2,
                "parking_zone": zone,
            }
        ]
    )

    clf = bundle["classifier"]
    reg = bundle["regressor"]
    proba = float(clf.predict_proba(row[FEATURE_COLUMNS + ["parking_zone"]])[0, 1])
    occ_hat = float(np.clip(reg.predict(row[FEATURE_COLUMNS + ["parking_zone"]])[0], 0, 1))

    m1, m2, m3 = st.columns(3)
    m1.metric("P(available)", f"{proba:.1%}")
    m2.metric("Predicted occupancy", f"{occ_hat:.1%}")
    m3.metric("Predicted free spaces", f"{int(round((1 - occ_hat) * capacity))}")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=occ_hat * 100,
            number={"suffix": "%"},
            title={"text": "Predicted fullness"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f6f54"},
                "steps": [
                    {"range": [0, OCCUPANCY_FULL_THRESHOLD * 100], "color": "#d7ebe2"},
                    {"range": [OCCUPANCY_FULL_THRESHOLD * 100, 100], "color": "#f3d6b5"},
                ],
                "threshold": {
                    "line": {"color": "#c47a2c", "width": 3},
                    "value": OCCUPANCY_FULL_THRESHOLD * 100,
                },
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(
        f"Classifier: **{bundle['meta']['best_classifier']}** · "
        f"Regressor: **{bundle['meta']['best_regressor']}**"
    )

elif page == "Melbourne live map":
    st.header("Melbourne on-street bay sensors")
    st.caption(
        "Live snapshot from City of Melbourne open data — separate from the Birmingham "
        "training set. Shows real occupied / unoccupied bay status on a map."
    )
    mel = load_melbourne()
    if mel is None or mel.empty:
        st.warning("Melbourne CSV missing. Run `python scripts/download_data.py --dataset melbourne`.")
        st.stop()

    avail_rate = mel["is_available"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensors in snapshot", f"{len(mel):,}")
    c2.metric("Unoccupied now", f"{avail_rate:.1%}")
    c3.metric("Occupied / other", f"{1 - avail_rate:.1%}")

    status_filter = st.multiselect(
        "Status filter",
        sorted(mel["status_clean"].unique()),
        default=[s for s in ["Unoccupied", "Present"] if s in set(mel["status_clean"])],
    )
    view = mel[mel["status_clean"].isin(status_filter)] if status_filter else mel
    color_map = {"Unoccupied": "#1f6f54", "Present": "#c47a2c", "Unknown": "#8a9a93"}
    fig = px.scatter_mapbox(
        view,
        lat="lat",
        lon="lon",
        color="status_clean",
        color_discrete_map=color_map,
        zoom=13,
        height=560,
        hover_data=["kerbsideid", "zone_number", "status_timestamp"],
        title="Melbourne CBD parking bay status",
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
<span class="footnote">
Source: <a href="https://data.melbourne.vic.gov.au/explore/dataset/on-street-parking-bay-sensors/">
City of Melbourne — On-street Parking Bay Sensors</a> (CC BY).
This page is exploratory context; ML models above are trained on Birmingham occupancy.
</span>
""",
        unsafe_allow_html=True,
    )
