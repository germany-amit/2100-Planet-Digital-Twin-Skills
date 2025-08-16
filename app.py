import streamlit as st
import pydeck as pdk
import requests
import pandas as pd
import numpy as np
import random, time
from collections import deque
from datetime import datetime, timezone

# ========================= Page Setup & Styles =========================
st.set_page_config(page_title="Future Digital Twin 2050–2100", page_icon="🤖", layout="wide")

st.markdown("""
<style>
footer {visibility:hidden;}
.block-container {padding-top: 0.75rem;}
.badge {display:inline-block;padding:.25rem .5rem;border-radius:9999px;background:#eef2ff;color:#3730a3;font-weight:600;margin-right:.5rem}
.panel {border:1px solid #e5e7eb;border-radius:18px;padding:1rem;background:white;box-shadow:0 6px 18px rgba(0,0,0,.05)}
.kpi {font-weight:700}
.thinking {font-variant-caps:all-small-caps;letter-spacing:.06em;opacity:.85}
.small {opacity:.75;font-size:.9rem}
.map-note {position:relative; top:-8px; opacity:.7}
</style>
""", unsafe_allow_html=True)

# ========================= Sidebar (Controls) =========================
with st.sidebar:
    st.header("⚙️ Controls")
    source = st.radio(
        "Live open data source:",
        [
            "USGS Earthquakes (24h)",    # many points with lat/lon + magnitude
            "OpenAQ PM2.5 (global latest)", # many points with lat/lon + pm2.5
            "ISS Current Position",      # single moving dot
            "COVID-19 Cases by Country", # many points with lat/lon + today cases
            "Sample Weather (6 cities)", # small set, multiple calls
        ],
        index=0,
        help="Switch if one source is temporarily down.",
    )

    st.divider()
    auto = st.toggle("🔁 Auto-refresh", value=True, help="Auto refreshes the dashboard at the chosen interval.")
    interval_sec = st.slider("Auto-refresh every (seconds)", 5, 60, 12)
    anomaly_z = st.slider("Anomaly threshold |z|", 1.5, 5.0, 3.0, 0.1)
    point_scale = st.slider("Map point scale", 10000, 200000, 70000, step=5000)
    use_heatmap = st.toggle("Heatmap for dense data", value=True)
    st.caption("Tip: Use Heatmap for Earthquakes/OpenAQ, Scatter for sparse data like ISS/Weather.")

    st.divider()
    st.markdown("**Back-end Brain Modules**")
    st.caption("AI/ML/DL • NLP/LLM • OCR/ICR • Data Science • Agentic AI • Optimization • Governance\n(visualized in status bars below)")

# Auto-refresh to make UX truly live
if auto:
    st_autorefresh = st.experimental_rerun  # fallback name if old Streamlit
    st.autorefresh(interval=interval_sec * 1000, key="autorefresh_key")

# ========================= Helpers =========================
def safe_get_json(url, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_usgs():
    """USGS earthquakes last 24h -> DataFrame(lat, lon, value, label)"""
    j = safe_get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
    feats = j.get("features", [])
    rows = []
    for f in feats:
        try:
            lon, lat, _ = f["geometry"]["coordinates"]
            mag = f["properties"]["mag"] or 0.0
            place = f["properties"]["place"] or "Unknown"
            rows.append({"lat": float(lat), "lon": float(lon), "value": float(mag), "label": f"M {mag:.1f} — {place}"})
        except Exception:
            continue
    return pd.DataFrame(rows)

def fetch_openaq():
    """OpenAQ latest PM2.5 -> DataFrame(lat, lon, value, label)"""
    j = safe_get_json("https://api.openaq.org/v2/latest?limit=200&parameter=pm25&order_by=measurements_value&sort=desc")
    rows = []
    for item in j.get("results", []):
        coords = item.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        # pick first measurement value
        meas = item.get("measurements", [])
        if not meas:
            continue
        val = meas[0].get("value")
        city = item.get("city") or item.get("location") or "Unknown"
        rows.append({"lat": float(lat), "lon": float(lon), "value": float(val), "label": f"PM2.5 {val} µg/m³ — {city}"})
    return pd.DataFrame(rows)

def fetch_iss():
    """ISS now -> DataFrame single point(lat, lon, value, label)"""
    j = safe_get_json("http://api.open-notify.org/iss-now.json")
    lat = float(j["iss_position"]["latitude"])
    lon = float(j["iss_position"]["longitude"])
    return pd.DataFrame([{"lat": lat, "lon": lon, "value": abs(lat), "label": f"ISS @ {lat:.2f}, {lon:.2f}"}])

def fetch_covid():
    """COVID by country -> DataFrame(lat, lon, value, label)"""
    j = safe_get_json("https://disease.sh/v3/covid-19/countries?yesterday=false&allowNull=true")
    rows = []
    for item in j:
        info = item.get("countryInfo") or {}
        lat, lon = info.get("lat"), info.get("long")
        if lat is None or lon is None:
            continue
        val = item.get("todayCases") or 0
        name = item.get("country") or "Unknown"
        rows.append({"lat": float(lat), "lon": float(lon), "value": float(val), "label": f"{name}: +{int(val)} today"})
    df = pd.DataFrame(rows)
    # keep a manageable sample for speed
    return df.sort_values("value", ascending=False).head(180)

def fetch_weather_sample():
    """Open-Meteo current weather for 6 cities -> DataFrame"""
    cities = [
        ("London", 51.5074, -0.1278),
        ("New York", 40.7128, -74.0060),
        ("Tokyo", 35.6762, 139.6503),
        ("Mumbai", 19.0760, 72.8777),
        ("Sydney", -33.8688, 151.2093),
        ("São Paulo", -23.5505, -46.6333),
    ]
    rows = []
    for name, lat, lon in cities:
        try:
            j = safe_get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
            cw = j.get("current_weather", {})
            temp = cw.get("temperature")
            ws = cw.get("windspeed")
            if temp is not None:
                rows.append({"lat": lat, "lon": lon, "value": float(temp), "label": f"{name}: {temp}°C, wind {ws} km/h"})
        except Exception:
            continue
    return pd.DataFrame(rows)

FETCHERS = {
    "USGS Earthquakes (24h)": fetch_usgs,
    "OpenAQ PM2.5 (global latest)": fetch_openaq,
    "ISS Current Position": fetch_iss,
    "COVID-19 Cases by Country": fetch_covid,
    "Sample Weather (6 cities)": fetch_weather_sample,
}

# per-source rolling buffers for Edge analytics (so z-score adapts over time)
if "buffers" not in st.session_state:
    st.session_state.buffers = {}
def get_buffer(key, maxlen=240):
    if key not in st.session_state.buffers:
        st.session_state.buffers[key] = deque(maxlen=maxlen)
    return st.session_state.buffers[key]

def compute_edge_stats(values: np.ndarray, buf: deque, zthr: float):
    """Update buffer with mean value and compute z for the latest mean."""
    if len(values) == 0:
        return None, None, None, "UNKNOWN"
    latest_signal = float(np.mean(values))
    buf.append(latest_signal)
    arr = np.array(buf, dtype=float)
    mu = float(np.mean(arr)) if len(arr) > 1 else latest_signal
    sd = float(np.std(arr)) if len(arr) > 1 else 1e-6
    z = (latest_signal - mu) / (sd if sd > 1e-6 else 1e-6)
    label = "ANOMALY" if abs(z) >= zthr else "NORMAL"
    return latest_signal, mu, z, label

def color_from_value_z(v, vmin, vmax, z, zthr):
    """
    Map value to blue->green gradient; override to red for anomalies.
    Return [r,g,b,alpha]
    """
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))
    # gradient (blue to green)
    r = int(30 + 20 * t)
    g = int(80 + 150 * t)
    b = int(180 - 120 * t)
    if abs(z) >= zthr:
        r, g, b = 220, 40, 40  # anomaly -> red
    return [r, g, b, 160]

def robot_policy(source_name: str, edge_label: str, df: pd.DataFrame):
    # Human-like rules
    if df.empty:
        return "🤖 Standing by. No data."
    if source_name.startswith("USGS"):
        high = df["value"].max() if "value" in df else 0
        if high >= 6.0:
            return "🛟 Seismic alert: notify sites, halt critical ops, dispatch inspection drones."
        elif high >= 4.5:
            return "🧯 Moderate quakes: run safety checks in affected radius."
        else:
            return "🧩 Low activity: continue monitoring."
    if source_name.startswith("OpenAQ"):
        pm = df["value"].median()
        if pm >= 150:
            return "😷 Poor air: advise masks, increase filtration & indoor operations."
        elif pm >= 75:
            return "🌀 Elevated PM2.5: optimize ventilation, notify H&S."
        else:
            return "🌿 Air acceptable: proceed normally."
    if source_name.startswith("ISS"):
        return "🛰 Adjust satcom windows & pass handoffs; log orbital pass."
    if source_name.startswith("COVID"):
        hot = (df["value"] >= 5000).sum()
        if hot > 10:
            return "🧪 Rising cases in many regions: tighten protocols, enable remote shifts."
        return "🏥 Health posture normal; keep monitoring."
    if source_name.startswith("Sample Weather"):
        hot = (df["value"] >= 40).sum()
        cold = (df["value"] <= -5).sum()
        if hot or cold:
            return "🌡 Extreme temps detected: adjust outdoor work schedules & HVAC loads."
        return "🌤 Weather normal across hubs."
    return "🤝 Continue operations with standard safety."

# ========================= Header (Animated Thinking) =========================
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("### 📡 IoT")
    dots = "." * (int(time.time()) % 3 + 1)
    st.markdown(f"<div class='thinking'>collecting sensors{dots}</div>", unsafe_allow_html=True)
with colB:
    st.markdown("### ⚡ Edge")
    dots = "." * ((int(time.time()) + 1) % 3 + 1)
    st.markdown(f"<div class='thinking'>analyzing stream{dots}</div>", unsafe_allow_html=True)
with colC:
    st.markdown("### 🤖 Robot")
    dots = "." * ((int(time.time()) + 2) % 3 + 1)
    st.markdown(f"<div class='thinking'>planning actions{dots}</div>", unsafe_allow_html=True)

st.markdown("")

# ========================= Main Panels =========================
map_col, kpi_col = st.columns([2, 1])

with map_col:
    st.markdown("#### 🌍 Live Digital Twin Map")
    # Fetch data with graceful fallback
    try:
        df = FETCHERS[source]()
    except Exception as e:
        st.error(f"Data source error: {e}")
        # fallback: random points (visible but marked as simulated)
        df = pd.DataFrame([{
            "lat": random.uniform(-60, 60),
            "lon": random.uniform(-180, 180),
            "value": random.random() * 10,
            "label": "Simulated point (fallback)"
        } for _ in range(50)])

    if df.empty:
        st.warning("No data returned. Try another source from the sidebar.")
    else:
        # Edge stats over time
        buf = get_buffer(source)
        latest_signal, mu, z, edge_label = compute_edge_stats(df["value"].to_numpy(), buf, anomaly_z)

        # Per-point colors (use global z for anomaly highlighting)
        vmin, vmax = float(df["value"].min()), float(df["value"].max())
        df["color"] = df.apply(lambda r: color_from_value_z(r["value"], vmin, vmax, z if z is not None else 0.0, anomaly_z), axis=1)

        # Choose layer type
        layers = []
        if use_heatmap and len(df) > 30:
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position='[lon, lat]',
                get_weight='value',
                aggregation='"SUM"'
            ))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[lon, lat]',
            get_radius=point_scale,
            get_fill_color='color',
            pickable=True
        ))

        initial_view = pdk.ViewState(
            latitude=float(df["lat"].mean()),
            longitude=float(df["lon"].mean()),
            zoom=1 if len(df) > 80 else 2.5,
            pitch=0
        )

        tooltip = {"text": "{label}\nValue: {value}"}
        deck = pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10", initial_view_state=initial_view, layers=layers, tooltip=tooltip)
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("<span class='map-note'>Colors: blue→green by value, red indicates **Edge anomaly**.</span>", unsafe_allow_html=True)

with kpi_col:
    st.markdown("#### 📊 Edge Analytics")
    if not df.empty:
        anomalies = int((np.abs((df["value"] - df["value"].mean()) / (df["value"].std() + 1e-6)) >= anomaly_z).sum())
        st.metric("Points", len(df))
        st.metric("Value range", f"{df['value'].min():.2f} – {df['value'].max():.2f}")
        st.metric("Anomalous points (local z)", anomalies)
        if "USGS" in source:
            st.metric("Strongest quake (M)", f"{df['value'].max():.2f}")
        elif "OpenAQ" in source:
            st.metric("Median PM2.5", f"{df['value'].median():.1f} µg/m³")
        elif "COVID" in source:
            st.metric("Total today (sample)", f"{int(df['value'].sum()):,}")
        elif "Sample Weather" in source:
            st.metric("Avg temperature", f"{df['value'].mean():.1f} °C")
        elif "ISS" in source:
            st.metric("Latitude |signal|", f"{df['value'].iloc[0]:.2f}")

    st.markdown("#### 🧠 Back-end Brain")
    modules = [
        ("AI/ML/DL", 70, "Scoring & thresholds"),
        ("NLP/LLM", 65, "Summaries/prompts"),
        ("OCR/ICR", 35, "Docs/forms (stub)"),
        ("Data Science", 80, "Features & sanity checks"),
        ("Agentic AI", 60, "Planning tools/chains"),
        ("Optimization", 55, "Cost/time routing"),
        ("Governance", 75, "Policy & safety"),
    ]
    for name, base, note in modules:
        p = st.progress(min(100, base + random.randint(-8, 8)))
        st.caption(f"**{name}** — {note}")

# ========================= Robot Decision & Log =========================
st.markdown("---")
log_col, action_col = st.columns([2, 1])

with action_col:
    st.markdown("### 🤖 Robot Action")
    if 'df' in locals() and not df.empty:
        action = robot_policy(source, "ANOMALY" if 'z' in locals() and abs(z) >= anomaly_z else "NORMAL", df)
        st.success(action)
    else:
        st.info("Waiting for valid data…")

with log_col:
    st.markdown("### 📜 Live Operations Log")
    box = st.empty()
    logs = st.session_state.get("logs", [])
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    logs.append(f"{now} — IoT fetched: {source}")
    if 'z' in locals() and z is not None:
        logs.append(f"{now} — Edge z-score: {z:.2f} (thr {anomaly_z:.1f})")
    if 'action' in locals():
        logs.append(f"{now} — Robot: {action}")
    st.session_state["logs"] = logs[-12:]  # keep last 12
    box.markdown("<div class='panel' style='height:190px;overflow:auto'>" + "<br>".join(st.session_state["logs"]) + "</div>", unsafe_allow_html=True)

# ========================= Skills Panel (Visible) =========================
st.markdown("---")
with st.expander("💡 30 Skills Demonstrated (visible & live)"):
    st.markdown("""
**Core AI/ML Skills** — Prompt Engineering; NLP Summarization; Foresight Modeling; Knowledge Graph Thinking; Generative AI Storytelling; Simulation Thinking; Decision Intelligence; Ethical AI Awareness; Multi-Agent Systems; AI Communication.  

**Robotics + IoT + Edge AI** — IoT Data Integration; Real-time Analytics; Digital Twin Thinking; Edge AI (TinyML concepts); Robotics Foresight; Automation Opportunity Discovery; Safety AI; Latency-Aware AI; AI for Resilience; AI + Sustainability.  

**Engineering & Data** — API Handling; Data Cleaning & Normalization; Visualization; Python Streamlit Engineering; Software Architecture (lightweight); Requirements Engineering; Error Handling; Scalable Design Thinking; Version Control / GitHub; Open Source Strategy.
""")

# ========================= Footer =========================
st.markdown("---")
st.caption("This free-tier app uses public APIs (no keys), live map updates, rolling Edge analytics, and human-like Robot decisions. Swap stubs with real endpoints/queues when moving to production.")
