import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import time

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="Global Operational Risk Radar", layout="wide")

st.title("🌍 Global Operational Risk Radar (2050-2100 Prototype)")
st.markdown("Live digital twin of enterprise sites • Seismic • Weather • Air Quality • Cyber Advisories • Unrest Signals")

# ------------------------------
# Sidebar: Inputs
# ------------------------------
st.sidebar.header("⚙️ Settings")

# Upload Sites CSV
sites_file = st.sidebar.file_uploader("Upload sites CSV (name,lat,lon)", type="csv")
if sites_file:
    sites = pd.read_csv(sites_file)
else:
    # demo sites
    sites = pd.DataFrame({
        "name": ["Delhi HQ", "NY Data Center", "Tokyo Hub", "Berlin Office"],
        "lat": [28.61, 40.71, 35.68, 52.52],
        "lon": [77.20, -74.00, 139.69, 13.40]
    })

# Choose Data Sources
st.sidebar.subheader("📡 Data Sources")
sources = st.sidebar.multiselect(
    "Select APIs",
    ["USGS Earthquakes", "Open-Meteo Weather", "OpenAQ PM2.5", "CISA KEV (Cyber)", "GDELT Unrest Signals"],
    default=["USGS Earthquakes", "Open-Meteo Weather"]
)

# ------------------------------
# Thinking Status
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🤖 Robot: *Planning actions…*")
with col2:
    st.warning("⚡ Edge: *Analyzing streams…*")
with col3:
    st.success("📡 IoT: *Collecting sensors…*")

st.markdown("---")

# ------------------------------
# Fetch Data (Demo APIs for free tier)
# ------------------------------

def get_usgs_eq():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        r = requests.get(url, timeout=5).json()
        return [(f["properties"]["place"], f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
                for f in r["features"][:10]]
    except:
        return []

def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        r = requests.get(url, timeout=5).json()
        return r.get("current", {}).get("temperature_2m", None)
    except:
        return None

eq_data = get_usgs_eq() if "USGS Earthquakes" in sources else []

# ------------------------------
# Map
# ------------------------------
st.subheader("🗺️ Live Risk Map")

m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

# Add enterprise sites
for _, row in sites.iterrows():
    risk_color = "green"
    if "Open-Meteo Weather" in sources:
        temp = get_weather(row["lat"], row["lon"])
        if temp and temp > 35:
            risk_color = "red"
        elif temp and temp > 25:
            risk_color = "orange"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color=risk_color,
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['name']} - Risk Level: {risk_color}"
    ).add_to(m)

# Add earthquakes
for place, lat, lon in eq_data:
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
        popup=f"Earthquake near {place}"
    ).add_to(m)

st_folium(m, width=900, height=500)

# ------------------------------
# KPIs
# ------------------------------
st.subheader("📊 KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Sites Monitored", len(sites))
col2.metric("Earthquakes (24h)", len(eq_data))
col3.metric("APIs Active", len(sources))

# ------------------------------
# Robot Recommended Actions
# ------------------------------
st.subheader("🤖 Robot Recommendations")
for _, row in sites.iterrows():
    st.write(f"✅ {row['name']}: Ensure backup power & cooling systems. Monitor local conditions.")
