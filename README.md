This repository contains a Streamlit application that simulates and visualizes real-time data for an electric vehicle (EV) fleet. The dashboard displays key metrics such as state-of-charge (SOC), voltage, current, speed, temperature, gradient, and the predicted range based on a machine learning model. It also integrates an interactive map showing EV charging stations in Kenya using Pydeck.

Features
Real-Time Data Streaming: Simulates live EV sensor data updates.
Predictive Modeling: Uses a RandomForestRegressor to estimate the EV's range.
Optimization Suggestions: Offers recommendations (e.g., advise charging, cool down battery, optimize speed) based on real-time data.
Interactive Map: Displays charging stations in Kenya sourced from a CSV file.
User-Friendly Interface: Controlled simulation via a sidebar button.
