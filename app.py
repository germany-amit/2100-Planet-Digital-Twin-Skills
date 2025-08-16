# ------------------------------------------------------------
# Fortune 500 Global Operational Risk Radar (Free-tier ready)
# - Works on Streamlit Cloud + GitHub (no API keys)
# - Live hazards: Earthquakes (USGS), Weather (Open-Meteo),
#   Air Quality (OpenAQ), Cyber Advisories (CISA KEV), Unrest Signals (GDELT)
# - Upload your sites CSV (name,lat,lon) or use demo locations
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
import time
from datetime import datetime, timezone, timedelta
import math
import io
import random

# ========================= Page & Styles =========================
st.set_page_config(page_title="Operational Risk Radar", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
footer {visibility:hidden;}
.block-container {padding-top:.75rem;}
.panel {border:1px solid #e5e7eb;border-radius:18px;padding:1rem;background:white;box-shadow:0 6px 18px rgba(0,0,0,.05)}
.thinking {font-variant-caps:all-small-caps;letter-spacing:.06em;opacity:.9}
.badge {display:inline-block;padding:.25rem .6rem;border-radius:9999px;background:#eef2ff;color:#3730a3;font-weight:600;margin-right:.5rem}
.small {opacity:.8}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Global Operational Risk Radar (2050-2100 Prototype)")
st.caption("Live digital twin of enterprise sites • Seismic • Weather • Air Quality • Cyber Advisories • Unrest Signals • Free-tier friendly")

# ========================= Sidebar Controls =========================
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**1) Sites**")
    uploaded = st.file_uploader("Upload sites CSV (name,lat,lon)", type=["csv"])
    st.caption("If not provided, demo sites will be used.")
    st.markdown("**2) Data Sources**")
    use_quakes = st.checkbox("USGS Earthquakes (24h)", True)
    use_weather = st.checkbox("Open-Meteo Weather", True)
    use_aq = st.checkbox("OpenAQ PM2.5", True)
    use_cisa = st.checkbox("CISA KEV (Cyber)", True)
    use_gdelt = st.checkbox("GDELT Unrest Signals", True)

    st.markdown("**3) Scoring Weights**")
    w_quake = st.slider("Weight: Seismic", 0.0, 1.0, 0.30, 0.05)
    w_weather = st.slider("Weight: Weather", 0.0, 1.0, 0.25, 0.05)
    w_aq = st.slider("Weight: Air Quality", 0.0, 1.0, 0.15, 0.05)
    w_cisa = st.slider("Weight: Cyber", 0.0, 1.0, 0.15, 0.05)
    w_gdelt = st.slider("Weight: Unrest", 0.0, 1.0, 0.15, 0.05)

    st.caption("Weights are normalized automatically.")

    st.markdown("**4) Refresh**")
    tick_btn = st.button("🔁 Refresh Now")
    st.caption("Auto-refresh add-on is optional; free tier works with manual refresh.")

# ========================= Demo Sites (if none uploaded) =========================
def demo_sites():
    return pd.DataFrame([
        {"name": "New York DC", "lat": 40.7128, "lon": -74.0060},
        {"name": "London DC", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo Plant", "lat": 35.6762, "lon": 139.6503},
        {"name": "Mumbai Hub", "lat": 19.0760, "lon": 72.8777},
        {"name": "Singapore Hub", "lat": 1.3521, "lon": 103.8198},
        {"name": "Frankfurt DC", "lat": 50.1109, "lon": 8.6821},
        {"name": "São Paulo Depot", "lat": -23.5505, "lon": -46.6333},
        {"name": "San Jose R&D", "lat": 37.3382, "lon": -121.8863},
    ])

if uploaded:
    try:
        sites = pd.read_csv(uploaded)
        if not set(["name","lat","lon"]).issubset(sites.columns):
            st.error("CSV must have columns: name, lat, lon")
            st.stop()
        sites = sites[["name","lat","lon"]].dropna()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        sites = demo_sites()
else:
    sites = demo_sites()

if sites.empty:
    st.error("No sites available.")
    st.stop()

# ========================= Utilities =========================
def safe_get_json(url, timeout=12):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ========================= Fetchers (No API keys) =========================
def fetch_quakes_24h():
    if not use_quakes: return []
    try:
        j = safe_get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
        feats = j.get("features", [])
        rows = []
        for f in feats:
            coords = f.get("geometry",{}).get("coordinates",[None,None])
            if coords and coords[0] is not None and coords[1] is not None:
                rows.append({
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "mag": float(f.get("properties",{}).get("mag") or 0.0),
                    "place": f.get("properties",{}).get("place","unknown")
                })
        return rows
    except Exception:
        return []

def fetch_weather_for(lat, lon):
    if not use_weather: return None
    try:
        j = safe_get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        cw = j.get("current_weather", {})
        return {
            "temp_c": float(cw.get("temperature")) if cw.get("temperature") is not None else None,
            "windspeed": float(cw.get("windspeed")) if cw.get("windspeed") is not None else None
        }
    except Exception:
        return None

def fetch_aq_pm25_near(lat, lon):
    if not use_aq: return None
    try:
        # Search within 50km for latest PM2.5
        j = safe_get_json(f"https://api.openaq.org/v2/latest?coordinates={lat},{lon}&radius=50000&parameter=pm25&limit=1&order_by=measurements_value&sort=desc")
        res = j.get("results", [])
        if not res: return None
        meas = res[0].get("measurements", [])
        if not meas: return None
        return float(meas[0].get("value"))
    except Exception:
        return None

def fetch_cisa_kev_recent_days(days=30):
    if not use_cisa: return 0
    try:
        j = safe_get_json("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        items = j.get("vulnerabilities", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cnt = 0
        for it in items:
            d = it.get("dateAdded")
            if not d: continue
            try:
                dt = datetime.fromisoformat(d.replace("Z","+00:00"))
            except Exception:
                continue
            if dt >= cutoff:
                cnt += 1
        return cnt
    except Exception:
        return 0

def fetch_gdelt_unrest_count(hours=24):
    if not use_gdelt: return 0
    try:
        # Query common unrest keywords globally over last 24h
        # (kept light to stay well within free tier)
        q = '("protest" OR "strike" OR "unrest" OR "riot")'
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={requests.utils.quote(q)}&format=json&mode=artlist&maxrecords=75&timespan={hours}h"
        j = safe_get_json(url)
        arts = j.get("articles", [])
        return len(arts)
    except Exception:
        return 0

# ========================= Animated “Thinking” =========================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📡 IoT")
    st.markdown(f"<div class='thinking'>collecting sensors{'.' * (int(time.time())%3+1)}</div>", unsafe_allow_html=True)
with col2:
    st.markdown("### ⚡ Edge")
    st.markdown(f"<div class='thinking'>analyzing streams{'.' * ((int(time.time())+1)%3+1)}</div>", unsafe_allow_html=True)
with col3:
    st.markdown("### 🤖 Robot")
    st.markdown(f"<div class='thinking'>planning actions{'.' * ((int(time.time())+2)%3+1)}</div>", unsafe_allow_html=True)

st.markdown("")

# ========================= Fetch Global Feeds Once =========================
with st.spinner("🔎 Fetching global hazard feeds…"):
    quakes = fetch_quakes_24h()
    kev_recent = fetch_cisa_kev_recent_days(days=30)
    unrest_24h = fetch_gdelt_unrest_count(hours=24)

# ========================= Per-site Enrichment =========================
def normalize_weights(ws):
    total = sum(ws)
    return [w/total if total>0 else 0 for w in ws]

wq, ww, wa, wc, wu = normalize_weights([w_quake, w_weather, w_aq, w_cisa, w_gdelt])

def near_quake_score(lat, lon, quakes):
    """Max magnitude within 500km scaled to 0..1 (cap mag at 8.0)"""
    if not quakes: return 0.0, None
    closest_mag = 0.0
    closest_info = None
    for q in quakes:
        d = haversine_km(lat, lon, q["lat"], q["lon"])
        if d <= 500:  # within 500 km
            if q["mag"] > closest_mag:
                closest_mag = q["mag"]
                closest_info = f'{q["mag"]:.1f} near {q["place"]}'
    score = min(closest_mag/8.0, 1.0)
    return score, closest_info

def weather_score(w):
    """Extreme heat/cold/wind -> 0..1"""
    if not w: return 0.0, None
    s = 0.0
    note = []
    t = w.get("temp_c")
    ws = w.get("windspeed")
    if t is not None:
        if t >= 40: s += 0.8; note.append(f"heat {t:.0f}°C")
        elif t >= 32: s += 0.5; note.append(f"warm {t:.0f}°C")
        elif t <= -5: s += 0.6; note.append(f"freeze {t:.0f}°C")
    if ws is not None:
        if ws >= 70: s = max(s, 0.8); note.append(f"wind {ws:.0f} km/h")
        elif ws >= 40: s = max(s, 0.5); note.append(f"wind {ws:.0f} km/h")
    return min(s,1.0), (", ".join(note) if note else None)

def aq_score(pm25):
    """PM2.5 to 0..1 using rough scale"""
    if pm25 is None: return 0.0, None
    # 0-50 good, 50-100 moderate, 100-150 unhealthy for SG, 150-250 unhealthy, >250 very unhealthy
    if pm25 <= 50: s = 0.1
    elif pm25 <= 100: s = 0.3
    elif pm25 <= 150: s = 0.5
    elif pm25 <= 250: s = 0.7
    else: s = 0.9
    return s, f"PM2.5 {pm25:.0f}"

def cyber_score(kev_count):
    """Recent KEV count -> coarse 0..1 (global pressure proxy)"""
    if kev_count >= 50: return 0.8, f"KEV +{kev_count}/30d"
    if kev_count >= 25: return 0.6, f"KEV +{kev_count}/30d"
    if kev_count >= 10: return 0.4, f"KEV +{kev_count}/30d"
    if kev_count >= 1:  return 0.2, f"KEV +{kev_count}/30d"
    return 0.0, "KEV stable"

def unrest_score(count):
    """Recent unrest mentions -> 0..1 (global pressure proxy)"""
    if count >= 60: return 0.8, f"Unrest {count}/24h"
    if count >= 30: return 0.6, f"Unrest {count}/24h"
    if count >= 10: return 0.4, f"Unrest {count}/24h"
    if count >= 1:  return 0.2, f"Unrest {count}/24h"
    return 0.0, "Unrest low"

rows = []
with st.spinner("🧩 Enriching sites with live data…"):
    for _, s in sites.iterrows():
        lat, lon = float(s["lat"]), float(s["lon"])

        # Weather
        w = fetch_weather_for(lat, lon) if use_weather else None
        ws, wnote = weather_score(w)

        # Air Quality
        pm = fetch_aq_pm25_near(lat, lon) if use_aq else None
        ascore, aqnote = aq_score(pm)

        # Quakes
        qs, qnote = near_quake_score(lat, lon, quakes) if use_quakes else (0.0, None)

        # Cyber (global)
        cs, cnote = cyber_score(kev_recent) if use_cisa else (0.0, None)

        # Unrest (global)
        us, unote = unrest_score(unrest_24h) if use_gdelt else (0.0, None)

        # Weighted risk
        total = wq*qs + ww*ws + wa*ascore + wc*cs + wu*us
        rows.append({
            "name": s["name"], "lat": lat, "lon": lon,
            "risk": round(total, 3),
            "quake": qs, "quake_note": qnote,
            "weather": ws, "weather_note": wnote,
            "air": ascore, "air_note": aqnote,
            "cyber": cs, "cyber_note": cnote,
            "unrest": us, "unrest_note": unote
        })

enriched = pd.DataFrame(rows)

# ========================= Robot Playbook =========================
def robot_recommendation(r):
    notes = []
    if r["quake"] >= 0.6: notes.append("🛟 Seismic checks, halt critical ops, inspection drones")
    elif r["quake"] >= 0.3: notes.append("🧯 Seismic precautionary inspection")

    if r["weather"] >= 0.6: notes.append("🌪 Adjust shifts & logistics, secure assets")
    elif r["weather"] >= 0.3: notes.append("🌡 Heat/cold plan; hydration/PPE")

    if r["air"] >= 0.7: notes.append("😷 N95s, filtration, indoor ops")
    elif r["air"] >= 0.4: notes.append("🌀 Ventilation optimization")

    if r["cyber"] >= 0.6: notes.append("🛡 Patch window now, MFA enforcement, isolate critical")
    elif r["cyber"] >= 0.3: notes.append("🔐 Heighten SOC monitoring")

    if r["unrest"] >= 0.6: notes.append("🚧 Route changes, travel freeze near hotspots")
    elif r["unrest"] >= 0.3: notes.append("🚦 Caution advisories to staff")

    if not notes:
        return "✅ Normal posture — continue operations"
    return " | ".join(notes)

enriched["robot_action"] = enriched.apply(robot_recommendation, axis=1)

# ========================= Map =========================
st.markdown("### 🌍 Live Risk Map")
if not enriched.empty:
    # color by risk (0..1): blue -> green -> orange -> red
    def color_from_risk(x):
        t = max(0.0, min(1.0, float(x)))
        if t < 0.33:    return [50, 160, 220, 200]
        elif t < 0.66:  return [80, 200, 120, 220]
        else:           return [220, 80, 60, 220]
    enriched["color"] = enriched["risk"].apply(color_from_risk)

    view = pdk.ViewState(latitude=float(enriched["lat"].mean()),
                         longitude=float(enriched["lon"].mean()),
                         zoom=1.6, pitch=0)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=enriched,
        get_position='[lon, lat]',
        get_radius=120000,
        get_fill_color='color',
        pickable=True,
    )
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view,
        layers=[layer],
        tooltip={"text": "{name}\nRisk: {risk}\n{robot_action}"}
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption("Colors indicate overall risk: blue/green (low) → orange/red (high). Tooltip shows the Robot’s recommended actions.")
else:
    st.info("No enriched data to map.")

# ========================= KPIs & Table =========================
kcol, tcol = st.columns([1,2])

with kcol:
    st.markdown("### 📊 KPIs")
    st.metric("Sites", len(enriched))
    st.metric("High Risk (≥0.66)", int((enriched["risk"]>=0.66).sum()))
    st.metric("Medium Risk (0.33–0.66)", int(((enriched["risk"]>=0.33)&(enriched["risk"]<0.66)).sum()))
    st.metric("Recent CISA KEV (30d)", int(fetch_cisa_kev_recent_days(30) if use_cisa else 0))
    st.metric("Unrest Mentions (24h)", int(fetch_gdelt_unrest_count(24) if use_gdelt else 0))

with tcol:
    st.markdown("### 🧭 Site Risk & Actions")
    show_cols = ["name","risk","quake_note","weather_note","air_note","cyber_note","unrest_note","robot_action"]
    st.dataframe(enriched[show_cols].sort_values("risk", ascending=False), use_container_width=True)

# ========================= Ops Log =========================
st.markdown("---")
log_col, action_col = st.columns([2,1])

with action_col:
    st.markdown("### 🤖 Robot (Decision Summary)")
    if len(enriched):
        worst = enriched.sort_values("risk", ascending=False).iloc[0]
        st.success(f"Top Priority: **{worst['name']}** — {worst['robot_action']}")
    else:
        st.info("Waiting for valid data…")

with log_col:
    st.markdown("### 📜 Live Operations Log")
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log_lines = [
        f"{now} — Fetched: Quakes={len(quakes) if use_quakes else 0}, CISA KEV(30d)={fetch_cisa_kev_recent_days(30) if use_cisa else 0}, Unrest(24h)={fetch_gdelt_unrest_count(24) if use_gdelt else 0}",
        f"{now} — Sites enriched: {len(enriched)}",
        f"{now} — Highest risk site: {worst['name']} ({worst['risk']:.2f})" if len(enriched) else f"{now} — No sites",
    ]
    if "opslog" not in st.session_state: st.session_state.opslog=[]
    st.session_state.opslog += log_lines
    st.session_state.opslog = st.session_state.opslog[-12:]
    st.markdown("<div class='panel' style='height:190px;overflow:auto'>" + "<br>".join(st.session_state.opslog) + "</div>", unsafe_allow_html=True)

# ========================= Skills =========================
st.markdown("---")
with st.expander("💡 30 Skills Demonstrated (visible & live)"):
    st.markdown("""
**Core AI/ML Skills** — Prompt Engineering; NLP Summarization; Foresight Modeling; Knowledge Graph Thinking; Generative AI Storytelling; Simulation Thinking; Decision Intelligence; Ethical AI Awareness; Multi-Agent Systems; AI Communication.  

**Robotics + IoT + Edge AI** — IoT Data Integration; Real-time Analytics; Digital Twin Thinking; Edge AI (TinyML concepts); Robotics Foresight; Automation Opportunity Discovery; Safety AI; Latency-Aware AI; AI for Resilience; AI + Sustainability.  

**Engineering & Data** — API Handling; Data Cleaning & Normalization; Visualization; Python Streamlit Engineering; Software Architecture (lightweight); Requirements Engineering; Error Handling; Scalable Design Thinking; Version Control/GitHub; Open Source Strategy.
""")

st.markdown("---")
st.caption("Free-tier build: public APIs only, no secrets. Replace scoring/feeds with internal telemetry (MQTT/Kafka/CMDB) for production.")
