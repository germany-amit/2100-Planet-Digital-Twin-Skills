import streamlit as st
import requests
import datetime

st.set_page_config(page_title="Future Digital Twin 2050–2100", layout="wide")

# Title
st.title("🤖 Future Digital Twin 2050–2100 🌐")
st.markdown("Robotics | IoT | Edge AI | Multi-Agent AI | Future Scenarios")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌐 IoT Data", "🤖 Robotics", "⚡ Edge AI", "🔮 Future Scenarios"])

# ---------------- TAB 1: IoT Data ----------------
with tab1:
    st.header("🌐 Real-time IoT Data")
    col1, col2 = st.columns(2)

    # Earthquake data (USGS API)
    with col1:
        st.subheader("🌋 Earthquakes (last 24h)")
        try:
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
            data = requests.get(url, timeout=10).json()
            quakes = data["features"][:5]
            for q in quakes:
                mag = q["properties"]["mag"]
                place = q["properties"]["place"]
                st.write(f"**Magnitude {mag}** – {place}")
        except:
            st.error("Could not load earthquake data.")

    # Weather (Open-Meteo API)
    with col2:
        st.subheader("☀️ Weather (Sample: London)")
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.12&current_weather=true"
            weather = requests.get(url, timeout=10).json()
            w = weather["current_weather"]
            st.metric("Temperature", f"{w['temperature']}°C")
            st.metric("Windspeed", f"{w['windspeed']} km/h")
        except:
            st.error("Could not load weather data.")

# ---------------- TAB 2: Robotics ----------------
with tab2:
    st.header("🤖 Robotics Showcase")
    st.write("""
    By 2050, robots will be:
    - Factory automation specialists
    - Healthcare assistants
    - Space explorers
    - Household helpers
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/ASIMO_2011.jpg", caption="Honda ASIMO – early humanoid robot", width=400)
    st.success("🔮 Demo Vision: Imagine humanoid + swarm robots collaborating across industries.")

# ---------------- TAB 3: Edge AI ----------------
with tab3:
    st.header("⚡ Edge AI (TinyML demo)")
    st.write("""
    Edge AI will allow devices to run intelligence **locally without cloud**.  
    Example: A smart sensor predicting anomalies on-device.
    """)
    # Simple demo: predict threshold
    temp = st.slider("📟 Simulated Sensor Temperature", 20, 100, 40)
    if temp > 70:
        st.error("⚠️ Edge AI Alert: Overheating detected!")
    else:
        st.success("✅ Normal operating condition.")

# ---------------- TAB 4: Future Scenarios ----------------
with tab4:
    st.header("🔮 Future Scenarios 2025 → 2050 → 2100")
    year = st.selectbox("Select a Year", [2025, 2050, 2100])

    if year == 2025:
        st.info("🌍 Early adoption of Edge AI + IoT in smart cities. Robotics mainly industrial.")
    elif year == 2050:
        st.success("🤖 Robots integrated into daily life. Edge AI everywhere. IoT powers digital twin of Earth.")
    elif year == 2100:
        st.warning("🚀 Human-robot cohabitation. AI governance. Planetary-scale IoT across Earth & space.")

    st.subheader("🧠 Multi-Agent AI Perspectives")
    st.write("Different 'agents' interpret the same data differently:")

    text = "By 2050, robotics, IoT, and edge computing will redefine industries and daily life."
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**👔 CEO Agent**")
        st.write("Sees cost savings, efficiency, new revenue streams.")
    with col2:
        st.markdown("**🔬 Scientist Agent**")
        st.write("Focuses on sustainability, research, ethical AI.")
    with col3:
        st.markdown("**🤖 Robot Agent**")
        st.write("Optimizes collaboration with humans, autonomy.")
    with col4:
        st.markdown("**🔮 Futurist Agent**")
        st.write("Sees 2100+ scenarios: human-AI symbiosis, interplanetary IoT.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Free APIs | Demo vision for Fortune 500 (2050–2100)")
