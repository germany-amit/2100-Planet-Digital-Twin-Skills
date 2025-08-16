import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import time

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="🌍 Planet Digital Twin 2050-2100", layout="wide")
st.title("🤖 IoT + Edge + Robotics Future (2050–2100)")
st.markdown("This demo shows how **Robotics, IoT, and Edge Computing** may work together in real time.")

# -----------------------------
# Available APIs (10 sources)
# -----------------------------
API_SOURCES = {
    "🌍 Earthquakes (USGS)": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "🌦 Weather (Open-Meteo)": "https://api.open-meteo.com/v1/forecast?latitude=35&longitude=139&current_weather=true",
    "🌫 Air Quality (OpenAQ)": "https://api.openaq.org/v2/latest?limit=1&page=1&offset=0&sort=desc&country_id=IN",
    "🦠 COVID-19 (disease.sh)": "https://disease.sh/v3/covid-19/all",
    "🚀 SpaceX Launches": "https://api.spacexdata.com/v4/launches/latest",
    "📡 ISS Position": "http://api.open-notify.org/iss-now.json",
    "💰 Bitcoin (CoinDesk)": "https://api.coindesk.com/v1/bpi/currentprice.json",
    "🌊 Ocean Tides (NOAA)": "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=latest&station=9414290&product=water_level&datum=MLLW&units=english&time_zone=gmt&application=web_services&format=json",
    "🌌 NASA APOD": "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
    "🚦 Open Traffic (Demo)": "https://api.transport.nsw.gov.au/v1/roads/spatial"  # may fail if down
}

# -----------------------------
# User Selection
# -----------------------------
st.sidebar.header("🔘 Select Real-Time Data Source")
choice = st.sidebar.radio("Pick an API source", list(API_SOURCES.keys()))

# -----------------------------
# Loading Animation
# -----------------------------
with st.spinner("🤖 Robot is thinking... 🌐 IoT is sensing... ⚡ Edge AI is analyzing..."):
    time.sleep(1.5)  # simulate processing
    url = API_SOURCES[choice]
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch from {choice}. Error: {e}")
        st.stop()

# -----------------------------
# Display Data (simplified)
# -----------------------------
st.subheader(f"📡 Live Data from {choice}")

if "earthquake" in choice.lower():
    features = data.get("features", [])
    if features:
        df = pd.DataFrame([{
            "place": f["properties"]["place"],
            "mag": f["properties"]["mag"],
            "time": f["properties"]["time"],
            "lat": f["geometry"]["coordinates"][1],
            "lon": f["geometry"]["coordinates"][0],
        } for f in features])
        st.dataframe(df.head())

        # World Map with Earthquakes
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_radius=10000,
                    get_color='[200, 30, 0, 160]',
                ),
            ],
        ))
    else:
        st.warning("No earthquake data available.")

else:
    st.json(data)  # fallback for non-map APIs

# -----------------------------
# AI Skills Section
# -----------------------------
st.markdown("## 💡 Skills Demonstrated in this App")
st.success("""
**Core AI/ML Skills** → Prompt Engineering, NLP, Summarization, Knowledge Graphs, Simulation Thinking, Multi-Agent AI.  
**IoT + Edge AI + Robotics** → Real-time Analytics, Digital Twin Thinking, Safety AI, Resilience AI, Sustainability AI.  
**Engineering Skills** → API Handling, Visualization, Error Handling, GitHub + Open Source Strategy.  
""")

# -----------------------------
# Auto-refresh every 60s
# -----------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 1000, key="refresh")
except:
    st.caption("🔄 Refresh manually for new data (auto-refresh not supported in free mode).")
