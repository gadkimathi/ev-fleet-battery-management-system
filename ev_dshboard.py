import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging
import time
import pydeck as pdk

# Reduce Streamlit logging level to avoid clutter
logging.getLogger("streamlit").setLevel(logging.ERROR)

# --- Initialize Streamlit app ---
st.set_page_config(page_title="EV Fleet Optimization Dashboard", page_icon="🚗", layout="wide")
st.title("Real-Time EV Fleet Optimization Dashboard")
st.markdown("This dashboard simulates EV fleet data, showing real-time predictions and optimization suggestions.")

# --- Map: Charging Stations in Kenya from CSV ---
st.subheader("Charging Stations in Kenya")

@st.cache_data
def load_charging_data():
    # Load your CSV file (adjust the file path if needed)
    df = pd.read_csv(r"C:\Users\user\Downloads\EV\Charging_stationKE1.csv")
    # Ensure that your CSV file includes the following columns:
    # "Station", "Plug Types", "Number of Chargers", "Latitude", "Longitude"
    return df

try:
    charging_stations_df = load_charging_data()
    
    # Create a Pydeck layer for the charging stations
    charging_layer = pdk.Layer(
        "ScatterplotLayer",
        data=charging_stations_df,
        get_position=["Longitude", "Latitude"],
        get_color=[200, 30, 0, 160],
        get_radius=3000,  # Adjust radius as needed
        pickable=True
    )

    # Center the view based on the mean coordinates of the stations
    view_state = pdk.ViewState(
        latitude=charging_stations_df["Latitude"].mean(),
        longitude=charging_stations_df["Longitude"].mean(),
        zoom=6,
        pitch=0
    )

    # Create the deck.gl map with a tooltip displaying station details
    r = pdk.Deck(
        layers=[charging_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Station:</b> {Station} <br/>"
                    "<b>Plug Types:</b> {Plug Types} <br/>"
                    "<b>Chargers:</b> {Number of Chargers}",
            "style": {"color": "white"}
        }
    )
    st.pydeck_chart(r)
    st.markdown("_Charging station data sourced from your CSV file._")
except Exception as e:
    st.error(f"Error loading charging stations data: {e}")

# --- Step 1: Generate synthetic training data for model ---
np.random.seed(42)
n_samples = 1000
soc = np.random.uniform(20, 100, n_samples)
voltage = soc * 0.36 + np.random.uniform(0.9, 1.1, n_samples)
current = np.random.uniform(5, 20, n_samples)
speed = np.random.uniform(10, 40, n_samples)
temperature = np.random.uniform(0, 40, n_samples)
gradient = np.random.uniform(0, 15, n_samples)
range_km = (soc * 0.5) - (current * 1.2) - (speed * 0.4) - (gradient * 0.3) + (40 - temperature) * 0.2
range_km = np.clip(range_km, 0, None)

data = pd.DataFrame({
    'soc': soc,
    'voltage': voltage,
    'current': current,
    'speed': speed,
    'temperature': temperature,
    'gradient': gradient,
    'range_km': range_km
})

X = data[['soc', 'voltage', 'current', 'speed', 'temperature', 'gradient']]
y = data['range_km']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Prediction function ---
def predict_range(soc, voltage, current, speed, temperature, gradient):
    input_data = pd.DataFrame({
        'soc': [soc],
        'voltage': [voltage],
        'current': [current],
        'speed': [speed],
        'temperature': [temperature],
        'gradient': [gradient]
    })
    predicted_range = model.predict(input_data)[0]
    return predicted_range

# --- Step 2: Initialize variables for SOC progression ---
current_soc = 100  # Start SOC at 100%

# --- Step 3: Simulated data generator for real-time streaming ---
def generate_simulated_data():
    global current_soc
    current_soc -= np.random.uniform(0.5, 1.5)
    if current_soc < 0:
        current_soc = 0  # Clamp SOC at 0
    
    voltage = current_soc * 0.36 + np.random.uniform(0.9, 1.1)
    current = np.random.uniform(5, 20)
    speed = np.random.uniform(10, 40)
    temperature = np.random.uniform(0, 40)
    gradient = np.random.uniform(0, 15)
    return current_soc, voltage, current, speed, temperature, gradient

# --- Step 4: Control simulation using a sidebar button ---
with st.sidebar:
    st.header("Simulation Control")
    start_button = st.button("Start Simulation")
    st.markdown("Use the **Start Simulation** button to begin data streaming.")

# --- Main Data Display ---
st.subheader("Real-Time Data Stream")
placeholder = st.empty()

# --- Step 5: Run Simulation ---
if start_button:
    try:
        while current_soc > 0:  # Continue until SOC reaches 0
            # Generate simulated sensor data
            soc, voltage, current, speed, temperature, gradient = generate_simulated_data()
            predicted_range = predict_range(soc, voltage, current, speed, temperature, gradient)

            # Determine optimization action
            action = "Normal Operation"
            if soc < 30:
                action = "Advise Charging Soon"
            elif soc > 80 and temperature > 35:
                action = "Cool Down Battery"
            elif speed > 35 and gradient > 10:
                action = "Optimize Speed for Efficiency"

            # Display data in Streamlit
            with placeholder.container():
                st.metric("🔋 SOC", f"{soc:.2f} %")
                st.metric("🔌 Voltage", f"{voltage:.2f} V")
                st.metric("⚡ Current", f"{current:.2f} A")
                st.metric("🚗 Speed", f"{speed:.2f} km/h")
                st.metric("🌡️ Temperature", f"{temperature:.2f} °C")
                st.metric("⛰️ Gradient", f"{gradient:.2f} °")
                st.metric("📏 Predicted Range", f"{predicted_range:.2f} km")

                # Display optimization suggestion with a color alert
                if action == "Advise Charging Soon":
                    st.warning(f"Optimization Suggestion: **{action}**")
                elif action == "Cool Down Battery":
                    st.error(f"Optimization Suggestion: **{action}**")
                elif action == "Optimize Speed for Efficiency":
                    st.info(f"Optimization Suggestion: **{action}**")
                else:
                    st.success(f"Optimization Suggestion: **{action}**")

            # Pause briefly to simulate real-time data update
            time.sleep(1)

    except Exception as e:
        st.error(f"An error occurred: {e}")
