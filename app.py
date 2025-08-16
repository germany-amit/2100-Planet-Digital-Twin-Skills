# Global Operational Risk Digital Twin (IoT → Edge → Robot)
# Free-tier friendly: no API keys, small deps, robust error handling.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timezone, timedelta
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Operational Risk Digital Twin", page_icon="🛰️", layout="wide")

# ------------------------- Styles -------------------------
st.markdown("""
<style>
.block-container {padding-top: .6rem;}
.kpi {font-size: .9rem; color: #64748b}
.panel {border:1px solid #e5e7eb;border-radius:16px;padding:12px 14px;background:white;box-shadow:0 8px 24px rgba(0,0,0,.05)}
.badge {display:inline-block;padding:.25rem .6rem;border-radius:9999px;background:#eef2ff;color:#3730a3;font-weight:600;margin-right:.5rem}
.thinking {font-variant-caps: all-small-caps; letter-spacing: .06em; opacity: .95}
hr {margin: .5rem 0 .7rem 0}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Fortune-500 Operational Risk Digital Twin (IoT → Edge → Robot)")
st.caption("APIs as IoT sensors • Edge anomaly detection & scoring • Robot playbooks • Free GitHub + Streamlit Cloud")

# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    up = st.file_uploader("Upload Sites CSV (name,lat,lon)", type=["csv"])
    st.caption("If omitted, demo sites load.")

    st.subheader("📡 Sensor APIs (as IoT)")
    use_quakes  = st.checkbox("USGS Earthquakes (24h)", True)
    use_weather = st.checkbox("Open-Meteo (per-site weather)", True)
    use_aq      = st.checkbox("OpenAQ PM2.5 (per-site air)", True)
    use_cisa    = st.checkbox("CISA KEV (global cyber)", True)
    use_gdelt   = st.checkbox("GDELT unrest signals (global)", True)

    st.subheader("🧮 Risk Weights (Edge)")
    w_quake  = st.slider("Seismic", 0.0, 1.0, 0.30, 0.05)
    w_weather= st.slider("Weather", 0.0, 1.0, 0.25, 0.05)
    w_air    = st.slider("Air Quality", 0.0, 1.0, 0.15, 0.05)
    w_cyber  = st.slider("Cyber", 0.0, 1.0, 0.15, 0.05)
    w_unrest = st.slider("Unrest", 0.0, 1.0, 0.15, 0.05)
    st.caption("Weights are normalized automatically.")

    st.subheader("🔁 Refresh")
    st.button("Refresh Now")  # manual refresh for free tier

# ------------------------- Sites -------------------------
def demo_sites():
    return pd.DataFrame([
        {"name":"Delhi HQ",      "lat":28.6139, "lon":77.2090},
        {"name":"NY DataCenter", "lat":40.7128, "lon":-74.0060},
        {"name":"Tokyo Plant",   "lat":35.6762, "lon":139.6503},
        {"name":"Berlin Office", "lat":52.5200, "lon":13.4050},
        {"name":"Singapore Hub", "lat":1.3521,  "lon":103.8198},
        {"name":"São Paulo DC",  "lat":-23.5505,"lon":-46.6333}
    ])

if up:
    try:
        sites = pd.read_csv(up)
        assert set(["name","lat","lon"]).issubset(sites.columns)
        sites = sites[["name","lat","lon"]].dropna()
    except Exception as e:
        st.error(f"CSV error: {e}")
        sites = demo_sites()
else:
    sites = demo_sites()

if sites.empty:
    st.stop()

# ------------------------- Helpers -------------------------
@st.cache_data(ttl=300)
def safe_get_json(url, timeout=12):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R=6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat-a_lat)
    dl   = math.radians(b_lon-a_lon)
    x = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(x))

def normalize_weights(lst):
    s = sum(lst)
    return [x/s if s>0 else 0 for x in lst]

wq, ww, wa, wc, wu = normalize_weights([w_quake, w_weather, w_air, w_cyber, w_unrest])

# ------------------------- IoT: Sensor APIs -------------------------
@st.cache_data(ttl=300)
def iot_quakes_24h():
    if not use_quakes: return []
    try:
        j = safe_get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
        rows=[]
        for f in j.get("features", []):
            c = f.get("geometry",{}).get("coordinates",[None,None])
            if c and c[0] is not None and c[1] is not None:
                rows.append({
                    "lat":c[1], "lon":c[0],
                    "mag": float(f.get("properties",{}).get("mag") or 0),
                    "place": f.get("properties",{}).get("place","")
                })
        return rows
    except Exception:
        return []

@st.cache_data(ttl=300)
def iot_weather(lat, lon):
    if not use_weather: return None
    try:
        j = safe_get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        cw = j.get("current_weather", {})
        return {
            "temp_c": cw.get("temperature"),
            "windspeed": cw.get("windspeed")
        }
    except Exception:
        return None

@st.cache_data(ttl=600)
def iot_air_pm25(lat, lon):
    if not use_aq: return None
    try:
        j = safe_get_json(f"https://api.openaq.org/v2/latest?coordinates={lat},{lon}&radius=50000&parameter=pm25&limit=1&order_by=measurements_value&sort=desc")
        res = j.get("results", [])
        if not res: return None
        meas = res[0].get("measurements", [])
        if not meas: return None
        return float(meas[0].get("value"))
    except Exception:
        return None

@st.cache_data(ttl=3600)
def iot_cisa_kev_count(days=30):
    if not use_cisa: return 0
    try:
        j = safe_get_json("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        items = j.get("vulnerabilities", [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cnt=0
        for it in items:
            d=it.get("dateAdded")
            if not d: continue
            try:
                dt=datetime.fromisoformat(d.replace("Z","+00:00"))
                if dt>=cutoff: cnt+=1
            except Exception:
                continue
        return cnt
    except Exception:
        return 0

@st.cache_data(ttl=900)
def iot_unrest_mentions(hours=24):
    if not use_gdelt: return 0
    try:
        q='("protest" OR "strike" OR "unrest" OR "riot")'
        url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={requests.utils.quote(q)}&format=json&mode=artlist&maxrecords=100&timespan={hours}h"
        j=safe_get_json(url)
        return len(j.get("articles", []))
    except Exception:
        return 0

# Fetch once
quakes = iot_quakes_24h()
kev_recent = iot_cisa_kev_count(30)
unrest_24h = iot_unrest_mentions(24)

# ------------------------- Edge: Anomaly & Risk -------------------------
def edge_seismic_score(lat, lon):
    """Max nearby magnitude (<=500km) scaled 0..1"""
    if not quakes: return 0.0, None
    best=0.0; note=None
    for q in quakes:
        d = haversine_km(lat, lon, q["lat"], q["lon"])
        if d<=500:
            if q["mag"]>best:
                best=q["mag"]; note=f"M{q['mag']:.1f} near {q['place']}"
    return min(best/8.0,1.0), note

def edge_weather_score(w):
    if not w: return 0.0, None
    s=0.0; notes=[]
    t=w.get("temp_c"); ws=w.get("windspeed")
    if t is not None:
        if t>=40: s=max(s,0.8); notes.append(f"heat {t:.0f}°C")
        elif t>=32: s=max(s,0.5); notes.append(f"warm {t:.0f}°C")
        elif t<=-5: s=max(s,0.6); notes.append(f"freeze {t:.0f}°C")
    if ws is not None:
        if ws>=70: s=max(s,0.8); notes.append(f"wind {ws:.0f} km/h")
        elif ws>=40: s=max(s,0.5); notes.append(f"wind {ws:.0f} km/h")
    return min(s,1.0), (", ".join(notes) if notes else None)

def edge_air_score(pm25):
    if pm25 is None: return 0.0, None
    if pm25<=50:  s=0.1; tag="Good"
    elif pm25<=100: s=0.3; tag="Moderate"
    elif pm25<=150: s=0.5; tag="Unhealthy (SG)"
    elif pm25<=250: s=0.7; tag="Unhealthy"
    else: s=0.9; tag="Very Unhealthy"
    return s, f"PM2.5 {pm25:.0f} ({tag})"

def edge_cyber_score(cnt):
    if cnt>=50:  return 0.8, f"KEV +{cnt}/30d"
    if cnt>=25:  return 0.6, f"KEV +{cnt}/30d"
    if cnt>=10:  return 0.4, f"KEV +{cnt}/30d"
    if cnt>=1:   return 0.2, f"KEV +{cnt}/30d"
    return 0.0, "KEV stable"

def edge_unrest_score(cnt):
    if cnt>=60:  return 0.8, f"Unrest {cnt}/24h"
    if cnt>=30:  return 0.6, f"Unrest {cnt}/24h"
    if cnt>=10:  return 0.4, f"Unrest {cnt}/24h"
    if cnt>=1:   return 0.2, f"Unrest {cnt}/24h"
    return 0.0, "Unrest low"

def risk_to_color(x):
    t=float(max(0.0, min(1.0, x)))
    if t<0.33:   return "green"
    elif t<0.66: return "orange"
    else:        return "red"

# Build enriched site table
rows=[]
for _, s in sites.iterrows():
    lat, lon = float(s["lat"]), float(s["lon"])

    w = iot_weather(lat, lon) if use_weather else None
    pm= iot_air_pm25(lat, lon) if use_aq else None
    qs, qnote = edge_seismic_score(lat, lon) if use_quakes else (0.0, None)
    ws, wnote = edge_weather_score(w) if use_weather else (0.0, None)
    ascore, anote = edge_air_score(pm) if use_aq else (0.0, None)
    cs, cnote = edge_cyber_score(kev_recent) if use_cisa else (0.0, None)
    us, unote = edge_unrest_score(unrest_24h) if use_gdelt else (0.0, None)

    total = wq*qs + ww*ws + wa*ascore + wc*cs + wu*us

    rows.append({
        "name": s["name"], "lat": lat, "lon": lon,
        "risk": round(total,3),
        "seismic": qs, "seismic_note": qnote,
        "weather": ws, "weather_note": wnote,
        "air": ascore, "air_note": anote,
        "cyber": cs, "cyber_note": cnote,
        "unrest": us, "unrest_note": unote,
        "temp_c": None if not w else w.get("temp_c"),
        "windspeed": None if not w else w.get("windspeed"),
        "pm25": pm
    })

enriched = pd.DataFrame(rows)

# ------------------------- Robot: Playbooks -------------------------
def robot_playbook(r):
    acts=[]
    if r["seismic"]>=0.6: acts.append("🛟 Seismic checks; pause critical ops; drone inspection")
    elif r["seismic"]>=0.3: acts.append("🧯 Post-tremor inspection")

    if r["weather"]>=0.6: acts.append("🌪 Severe weather posture; secure assets; power backup")
    elif r["weather"]>=0.3: acts.append("🌡 Heat/cold plan; hydration/PPE")

    if r["air"]>=0.7: acts.append("😷 Enforce N95; indoor shift; filtration")
    elif r["air"]>=0.4: acts.append("🌀 Improve ventilation; monitor AQ")

    if r["cyber"]>=0.6: acts.append("🛡 Patch now; MFA; network segmentation")
    elif r["cyber"]>=0.3: acts.append("🔐 Heighten SOC monitoring")

    if r["unrest"]>=0.6: acts.append("🚧 Reroute logistics; travel freeze")
    elif r["unrest"]>=0.3: acts.append("🚦 Staff advisories; security briefing")

    return " | ".join(acts) if acts else "✅ Normal posture — continue operations"

enriched["robot_action"] = enriched.apply(robot_playbook, axis=1)

# ------------------------- Thinking header -------------------------
c1,c2,c3 = st.columns(3)
with c1: st.info(f"📡 IoT — sensing live…  \n{len(quakes) if use_quakes else 0} quakes • KEV(30d) {kev_recent if use_cisa else 0} • Unrest(24h) {unrest_24h if use_gdelt else 0}")
with c2: st.warning("⚡ Edge — scoring & anomaly detection running…")
with c3: st.success("🤖 Robot — generating site playbooks…")

st.markdown("")

# ------------------------- Map (folium) -------------------------
st.subheader("🗺️ Live Risk Map")
m = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB positron")

# Site markers (color by risk)
for _, r in enriched.iterrows():
    color = risk_to_color(r["risk"])
    popup = (f"<b>{r['name']}</b><br>"
             f"Risk: {r['risk']:.2f}<br>"
             f"{r['seismic_note'] or ''}<br>"
             f"{r['weather_note'] or ''}<br>"
             f"{r['air_note'] or ''}<br>"
             f"{r['cyber_note'] or ''}<br>"
             f"{r['unrest_note'] or ''}<br>"
             f"<i>{r['robot_action']}</i>")
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=9, color=color, fill=True, fill_opacity=0.85,
        popup=folium.Popup(popup, max_width=280)
    ).add_to(m)

# Optional quake pins (as sensor points)
if use_quakes and quakes:
    for q in quakes[:50]:
        folium.CircleMarker(
            location=[q["lat"], q["lon"]],
            radius=4, color="red", fill=True, fill_opacity=0.6,
            tooltip=f"M{q['mag']:.1f} {q['place']}"
        ).add_to(m)

st_folium(m, width=1000, height=420)

# ------------------------- Executive KPIs -------------------------
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Sites", len(enriched))
k2.metric("High Risk (≥0.66)", int((enriched["risk"]>=0.66).sum()))
k3.metric("Medium Risk (0.33–0.66)", int(((enriched["risk"]>=0.33)&(enriched["risk"]<0.66)).sum()))
k4.metric("CISA KEV (30d)", kev_recent if use_cisa else 0)
k5.metric("Unrest Mentions (24h)", unrest_24h if use_gdelt else 0)
st.caption("KPIs reflect current IoT feeds (via public APIs) analyzed on the Edge layer.")

# ------------------------- IoT/Edge/Robo Summaries per site -------------------------
st.subheader("🧭 Site Risk, IoT Summaries & Robot Actions")
show = enriched.copy()
show["IoT (Weather)"] = show.apply(lambda r: f"{'' if r['temp_c'] is None else str(round(r['temp_c']))+'°C'}, {'' if r['windspeed'] is None else str(round(r['windspeed']))+' km/h'}", axis=1)
show["IoT (Air)"] = show.apply(lambda r: f"{'' if r['pm25'] is None else str(int(r['pm25']))} μg/m³", axis=1)
show = show[["name","risk","seismic_note","IoT (Weather)","air_note","cyber_note","unrest_note","robot_action"]].sort_values("risk", ascending=False)
st.dataframe(show, use_container_width=True)

# ------------------------- Robot Priority Summary -------------------------
st.subheader("🤖 Robot: Priority Playbook")
if len(enriched):
    top = enriched.sort_values("risk", ascending=False).iloc[:3]
    for _, r in top.iterrows():
        st.markdown(f"**{r['name']} — Risk {r['risk']:.2f}**  \n{r['robot_action']}")

# ------------------------- Download report -------------------------
st.markdown("### ⬇️ Export Current Risk Report")
csv = enriched[["name","lat","lon","risk","seismic_note","weather_note","air_note","cyber_note","unrest_note","robot_action"]].sort_values("risk", ascending=False).to_csv(index=False)
st.download_button("Download CSV", csv, file_name="risk_report.csv", mime="text/csv")

st.markdown("---")
with st.expander("💡 What makes this Fortune-500 ready?"):
    st.markdown("""
- **Direct IoT Use**: Treats public APIs as **sensor feeds** (seismic, weather, air, cyber, unrest).
- **Edge Analytics**: On-the-fly anomaly detection, normalization & **weighted risk scoring** with transparent notes.
- **Robot Playbooks**: Site-specific, explainable **actions** that map to ops runbooks.
- **Exec Map + KPIs**: One-glance situational awareness; CSV export for audit.
- **Enterprise Path**: Replace public APIs with MQTT/Kafka/internal CMDB; wire playbooks to Jira/ServiceNow for tickets.
""")
