import streamlit as st
import requests
import numpy as np
import pandas as pd
import random, time
from datetime import datetime, timezone
from collections import deque

# ============= Page config =============
st.set_page_config(page_title="Future Digital Twin 2050–2100", page_icon="🤖", layout="wide")
st.markdown("""
<style>
footer {visibility:hidden;}
.small {opacity:.75;font-size:.9rem}
.kpi   {font-weight:700}
.logbox{height:190px;overflow:auto;border:1px solid #eee;border-radius:10px;padding:.75rem;background:#fafafa}
.card  {border:1px solid #eee;border-radius:16px;padding:1rem;background:white}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Future Digital Twin 2050–2100")
st.caption("Robotics • IoT • Edge Computing • Agentic AI • Real-time demo on free Streamlit + GitHub")

# ============= Sidebar =============
with st.sidebar:
    st.header("Data Source (IoT-like)")
    st.caption("Pick one; switch if an API is down.")
    source = st.radio(
        "Live Open APIs (no key):",
        [
            "USGS Earthquakes (24h)",
            "Open-Meteo Weather (London)",
            "Open-Meteo Air Quality (London)",
            "ISS Current Position",
            "Open-Notify Astronauts",
            "SpaceX Latest Launch",
            "CoinDesk BTC Price",
            "Disease.sh Global COVID",
            "WorldTime UTC",
            "Bored Activity",
            "Cat Fact (length proxy)",
            "Public Holidays (UK)",
        ],
        index=0
    )
    st.divider()
    st.caption("This demo simulates a realistic backend: AI/ML/DL • NLP/LLM • OCR/ICR • Data Science • Agentic AI • Optimization • Governance.")
    st.caption("All ‘thinking’ is performed in-session to fit free tier. Replace st.sleep with real jobs later.")

# ============= Helpers =============
def log(p, text):
    p.write(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} — {text}")

def safe_get(url, timeout=10, as_json=True):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json() if as_json else r.text

def fetch_signal(src: str):
    """
    Returns (signal_value: float|None, summary: str, context: dict)
    signal_value is a numeric indicator Edge will use (z-score/anomaly).
    """
    try:
        if src == "USGS Earthquakes (24h)":
            data = safe_get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
            feats = data.get("features", [])
            mags = [f["properties"].get("mag") for f in feats if f.get("properties")]
            mags = [m for m in mags if isinstance(m, (int, float))]
            sig = max(mags) if mags else 0.0
            return sig, f"Max magnitude in last 24h: {sig:.2f}", {"count": len(mags)}
        if src == "Open-Meteo Weather (London)":
            d = safe_get("https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.12&current_weather=true")
            cw = d.get("current_weather", {})
            temp = cw.get("temperature", None)
            wind = cw.get("windspeed", None)
            return float(temp) if temp is not None else None, f"Temp {temp}°C, Wind {wind} km/h", cw
        if src == "Open-Meteo Air Quality (London)":
            d = safe_get("https://air-quality-api.open-meteo.com/v1/air-quality?latitude=51.5&longitude=-0.12&hourly=pm2_5&forecast_days=1")
            pm25 = d.get("hourly", {}).get("pm2_5", [])
            sig = float(pm25[-1]) if pm25 else None
            return sig, f"PM2.5 latest: {sig} µg/m³", {}
        if src == "ISS Current Position":
            d = safe_get("http://api.open-notify.org/iss-now.json")
            lat = float(d["iss_position"]["latitude"])
            lon = float(d["iss_position"]["longitude"])
            # Use absolute latitude as a signal proxy (0..90)
            return abs(lat), f"ISS at lat {lat:.2f}, lon {lon:.2f}", {"lat": lat, "lon": lon}
        if src == "Open-Notify Astronauts":
            d = safe_get("http://api.open-notify.org/astros.json")
            n = int(d.get("number", 0))
            return float(n), f"Astronauts in space: {n}", {}
        if src == "SpaceX Latest Launch":
            d = safe_get("https://api.spacexdata.com/v4/launches/latest")
            success = d.get("success")
            name = d.get("name")
            sig = 100.0 if success else 10.0 if success is False else 50.0
            return sig, f"Latest launch: {name}, success={success}", {}
        if src == "CoinDesk BTC Price":
            d = safe_get("https://api.coindesk.com/v1/bpi/currentprice.json")
            usd = float(d["bpi"]["USD"]["rate_float"])
            return usd, f"BTC/USD: {usd:.2f}", {}
        if src == "Disease.sh Global COVID":
            d = safe_get("https://disease.sh/v3/covid-19/all")
            cases = float(d.get("todayCases", 0))
            return cases, f"Global new cases today: {int(cases)}", {}
        if src == "WorldTime UTC":
            d = safe_get("https://worldtimeapi.org/api/timezone/Etc/UTC")
            unixt = float(d.get("unixtime", 0))
            return unixt % 86400, "Seconds since UTC midnight (signal): {:.0f}".format(unixt % 86400), {}
        if src == "Bored Activity":
            d = safe_get("https://www.boredapi.com/api/activity")
            act = d.get("activity", "")
            # Use activity length as signal
            return float(len(act)), f"Activity: {act}", {}
        if src == "Cat Fact (length proxy)":
            d = safe_get("https://catfact.ninja/fact")
            fact = d.get("fact", "")
            return float(len(fact)), f"Cat fact len={len(fact)}", {}
        if src == "Public Holidays (UK)":
            d = safe_get("https://date.nager.at/api/v3/PublicHolidays/2025/GB")
            # Use number of remaining holidays in year as signal
            year = 2025
            remaining = len([h for h in d if h.get("date", "").startswith(str(year))])
            return float(remaining), f"UK holidays {year}: {remaining}", {}
    except Exception as e:
        return None, f"Fetch error: {e}", {}
    return None, "Unknown source", {}

def edge_analyze(signal_series: deque, new_value: float, win: int = 120):
    """Update rolling stats, compute z-score, and classify."""
    signal_series.append(new_value)
    arr = np.array(signal_series, dtype=float)
    mu = float(np.mean(arr)) if len(arr) >= 2 else new_value
    sd = float(np.std(arr)) if len(arr) >= 2 else 1e-6
    z = (new_value - mu) / (sd if sd > 1e-6 else 1e-6)
    # simple classification
    if abs(z) >= 3.0:
        label = "ANOMALY"
    elif abs(z) >= 2.0:
        label = "WATCH"
    else:
        label = "NORMAL"
    return mu, sd, z, label

def robot_decide(src: str, value, label: str):
    """Human-like policy for the robot based on source and label."""
    if value is None:
        return "🤖 Standing by: no valid data."
    if src.startswith("USGS"):
        if value >= 6.0: return "🛟 Evacuate & trigger emergency comms. Dispatch inspection drones."
        if value >= 4.5: return "🧯 Safety checklists to sites within 500 km. Pause non-critical ops."
        return "🧩 Continue monitoring seismic telemetry."
    if "Air Quality" in src:
        if value >= 150: return "😷 Advise masks/indoor ops. Increase HVAC filtration."
        if value >= 75:  return "🌀 Optimize ventilation. Notify health & safety."
        return "🌿 Air quality acceptable. Proceed."
    if "Weather" in src:
        if value >= 40:  return "❄️ Activate cooling protocols. Reschedule outdoor tasks."
        if value <= -5:  return "🔥 Activate heating & anti-ice measures."
        return "🌤 Normal weather window."
    if "BTC" in src:
        return "📈 Update risk model & treasury hedges."
    if "SpaceX" in src:
        return "🚀 Log outcome; update reliability priors."
    if "ISS" in src:
        return "🛰 Adjust satcom handoff windows."
    if "COVID" in src:
        if value >= 100000: return "🧪 Tighten on-site protocols; enable remote shifts."
        elif value >= 10000: return "🧼 Increase hygiene cadence; monitor closely."
        return "🏥 Normal health posture."
    if "Holidays" in src:
        return "📅 Adjust staffing & logistics for holiday calendar."
    return "🤝 Continue operations with standard safety."

# Session state buffers for Edge rolling stats per source
if "buffers" not in st.session_state:
    st.session_state.buffers = {}

def get_buffer(key, maxlen=120):
    if key not in st.session_state.buffers:
        st.session_state.buffers[key] = deque(maxlen=maxlen)
    return st.session_state.buffers[key]

# ============= Main 3-panels with "thinking" UX =============
col_iot, col_edge, col_robot = st.columns(3)

with col_iot:
    st.markdown("### 🌐 IoT — **thinking…**")
    iot_card = st.container()
    with iot_card:
        with st.spinner("Fetching live data from selected source… Please wait"):
            sig, summary, ctx = fetch_signal(source)
            st.markdown(f"**Source:** {source}")
            if sig is not None:
                st.metric("Signal", f"{sig:,.2f}")
            st.write(summary)
    st.caption("If a source fails, switch via sidebar. These APIs are public & free.")

with col_edge:
    st.markdown("### ⚡ Edge — **analyzing…**")
    edge_card = st.container()
    with edge_card:
        with st.spinner("Running on-device analytics (sliding stats, z-score)…"):
            buf = get_buffer(source)
            # seed with small noise if empty to stabilize std
            if len(buf) == 0 and isinstance(sig, (int, float)):
                for _ in range(30):
                    buf.append(float(sig) + np.random.normal(0, 0.1))
            if isinstance(sig, (int, float)):
                mu, sd, z, label = edge_analyze(buf, float(sig))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest", f"{sig:,.2f}")
                c2.metric("Mean", f"{mu:,.2f}")
                c3.metric("Std", f"{sd:,.2f}")
                c4.metric("z", f"{z:0.2f}")
                bar = st.progress(min(100, int(min(abs(z)/4*100, 100))))
                st.write(f"Edge classification: **{label}**")
            else:
                label = "UNKNOWN"
                st.warning("No numeric signal available to analyze.")

with col_robot:
    st.markdown("### 🤖 Robot — **reasoning like a human…**")
    robot_card = st.container()
    with robot_card:
        status = st.empty()
        logbox = st.container()
        with st.spinner("Planning actions with multi-skill pipeline…"):
            # Simulated backend pipeline (AI/ML/DL, NLP/LLM, OCR/ICR, Data Science, Agentic AI, Governance)
            # 1) Data science quick checks
            status.info("📊 Data Science: sanity checks & feature prep")
            time.sleep(0.2)
            # 2) ML quick inference (simulated)
            status.info("🤖 ML/DL: running lightweight inference & risk scoring")
            time.sleep(0.2)
            # 3) NLP/LLM prompt (simulated)
            status.info("🧠 NLP/LLM: drafting operator summary & recommendations")
            time.sleep(0.2)
            # 4) OCR/ICR (placeholder)
            status.info("🧾 OCR/ICR: scanning incoming docs (placeholder)")
            time.sleep(0.15)
            # 5) Agentic planner
            status.info("🧩 Agentic AI: evaluating tool options & costs")
            time.sleep(0.2)
            # 6) Governance
            status.info("⚖️ Governance: policy & safety checks")
            time.sleep(0.15)
            # Final decision
            action = robot_decide(source, sig, label)
        status.success("✅ Action planned")
        st.success(action)

# ============= Operations Log (feels alive) =============
st.markdown("---")
st.subheader("📜 Live Operations Log (simulated)")
logcol1, logcol2 = st.columns([2,1])
with logcol1:
    ph_log = st.empty()
    with ph_log.container():
        log_area = st.container()
        with log_area:
            st.markdown("<div class='logbox' id='logbox'>", unsafe_allow_html=True)
            lp = st.empty()
            logs = []
            def append_log(msg): 
                logs.append(f"{datetime.utcnow().strftime('%H:%M:%S')} — {msg}")
                lp.markdown("<br>".join(logs[-10:]), unsafe_allow_html=True)
            append_log(f"IoT fetched: {source}")
            if isinstance(sig, (int, float)):
                append_log(f"Edge stats updated (z computed).")
                append_log(f"Robot: {action}")
            else:
                append_log("Edge skipped; non-numeric signal.")
            st.markdown("</div>", unsafe_allow_html=True)
with logcol2:
    st.markdown("#### 🧠 Backend ‘Brain’ Modules")
    modules = [
        ("AI/ML/DL", 70, "Model scoring & thresholds"),
        ("NLP/LLM", 60, "Summaries, prompts, policies"),
        ("OCR/ICR", 40, "Docs & forms (placeholder)"),
        ("Data Science", 80, "Features & stats"),
        ("Agentic AI", 65, "Plan/decide tools"),
        ("Optimization", 55, "Cost/time routing"),
        ("Governance", 75, "Policy, safety, audit")
    ]
    for name, base, note in modules:
        prog = st.progress(min(100, base + random.randint(-10, 10)))
        st.caption(f"**{name}** — {note}")

# ============= Skills Panel =============
st.markdown("---")
with st.expander("💡 30 Skills Demonstrated (visible)"):
    st.write("""
**Core AI/ML Skills** — Prompt Engineering; NLP Summarization; Foresight Modeling; Knowledge Graph Thinking; Generative AI Storytelling; Simulation Thinking; Decision Intelligence; Ethical AI Awareness; Multi-Agent Systems; AI Communication.

**Robotics + IoT + Edge** — IoT Data Integration; Real-time Analytics; Digital Twin Thinking; Edge AI (TinyML concepts); Robotics Foresight; Automation Opportunity Discovery; Safety AI; Latency-Aware AI; AI for Resilience; AI + Sustainability.

**Engineering & Data** — API Handling; Data Cleaning & Normalization; Visualization; Python Streamlit Engineering; Software Architecture (lightweight); Requirements Engineering; Error Handling; Scalable Design Thinking; Version Control/GitHub; Open Source Strategy.
""")
st.caption("All ‘thinking’ UIs above map to these skills so Fortune 500 stakeholders *feel* the backend operating in real time.")

# ============= Footer =============
st.markdown("---")
st.subheader("How this maps to 2050–2100")
st.write("""
- **IoT:** multi-sensor world (cities, factories, orbit) with robust source switching.  
- **Edge:** local autonomy & privacy-first analytics (rolling stats, fast decisions).  
- **Robotics:** human-like agents executing safety-first playbooks with governance.  
- **Enterprise:** replace simulators with real jobs, queues, and device twins; wire to ROS, MQTT, Kafka, PLCs.
""")
st.caption("This free-tier app uses public APIs + simulated pipelines. Swap the simulated blocks with real endpoints as you scale.")
