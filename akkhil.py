import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

st.set_page_config(page_title="Earthquake Prediction", layout="wide")
st.title("🌍 Earthquake Prediction Using ML")
st.warning("Educational use only. This cannot predict future earthquakes reliably.")

FEATURES = ["latitude", "longitude", "depth"]
MODEL_FILE = "earthquake_model.joblib"

# --- NEW: Country filter ---
COUNTRIES = {
    "All Countries": None,
    "India": {"min_lat": 6.7, "max_lat": 37.1, "min_lon": 68.1, "max_lon": 97.4},
    "Japan": {"min_lat": 30.0, "max_lat": 46.0, "min_lon": 129.0, "max_lon": 146.0},
    "USA": {"min_lat": 24.0, "max_lat": 50.0, "min_lon": -125.0, "max_lon": -66.0},
    "Indonesia": {"min_lat": -11.0, "max_lat": 6.0, "min_lon": 95.0, "max_lon": 141.0},
    "Turkey": {"min_lat": 36.0, "max_lat": 42.0, "min_lon": 26.0, "max_lon": 45.0},
}

if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All Countries"

st.subheader("Select Country")
cols = st.columns(len(COUNTRIES))
for col, country in zip(cols, COUNTRIES.keys()):
    if col.button(country, use_container_width=True):
        st.session_state.selected_country = country

selected_country = st.session_state.selected_country
st.info(f"Selected Country: **{selected_country}**")

def load_data():
    uploaded = st.sidebar.file_uploader("Upload earthquake CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded)

    if os.path.exists("earthquakes.csv"):
        return pd.read_csv("earthquakes.csv")

    st.sidebar.info("No CSV found. Download recent data from USGS.")
    if st.sidebar.button("Download USGS data"):
        # If India is selected, download only India area data
        params = {
            "format": "geojson",
            "starttime": "2020-01-01",
            "minmagnitude": 2.5,
            "limit": 2000
        }
        bounds = COUNTRIES.get(selected_country)
        if bounds:
            params.update({
                "minlatitude": bounds["min_lat"],
                "maxlatitude": bounds["max_lat"],
                "minlongitude": bounds["min_lon"],
                "maxlongitude": bounds["max_lon"],
            })

        response = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params=params,
            timeout=40
        )
        response.raise_for_status()
        rows = []
        for event in response.json().get("features", []):
            p = event.get("properties") or {}
            coords = (event.get("geometry") or {}).get("coordinates", [])
            if len(coords) >= 3 and p.get("mag") is not None:
                rows.append({
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "depth": coords[2],
                    "magnitude": p["mag"]
                })
        return pd.DataFrame(rows)
    return None

try:
    df = load_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

if df is None:
    st.stop()

df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    "lat": "latitude", "lon": "longitude",
    "mag": "magnitude", "depth_km": "depth"
})

required = FEATURES + ["magnitude"]
if not all(c in df.columns for c in required):
    st.error("CSV must contain latitude, longitude, depth, magnitude.")
    st.stop()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required)
df = df[df["latitude"].between(-90, 90)]
df = df[df["longitude"].between(-180, 180)]
df = df[df["magnitude"].between(-2, 10)]

# --- NEW: Filter by selected country ---
bounds = COUNTRIES.get(selected_country)
if bounds:
    df = df[
        df["latitude"].between(bounds["min_lat"], bounds["max_lat"]) &
        df["longitude"].between(bounds["min_lon"], bounds["max_lon"])
    ]

if len(df) < 20:
    st.error(f"At least 20 valid records are required for {selected_country}. Upload a bigger CSV or download USGS data for {selected_country}.")
    st.stop()

st.success(f"Loaded {len(df)} earthquake records for {selected_country}.")

# --- NEW: Show ALL records for selected country ---
st.subheader(f"All Records - {selected_country}")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download Filtered CSV",
    df.to_csv(index=False),
    file_name=f"{selected_country.lower().replace(' ', '_')}_earthquakes.csv",
    mime="text/csv"
)

st.subheader("Earthquake Map")
st.map(df[["latitude", "longitude"]].rename(
    columns={"latitude": "lat", "longitude": "lon"}
))

fig, ax = plt.subplots()
df["magnitude"].hist(bins=25, ax=ax)
ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

if st.button("Train Model"):
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(train[FEATURES], train["magnitude"])
    predictions = model.predict(test[FEATURES])

    st.metric("MAE", f"{mean_absolute_error(test['magnitude'], predictions):.3f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(test['magnitude'], predictions)):.3f}")

    joblib.dump(model, MODEL_FILE)
    st.session_state["model"] = model
    st.success("Model trained and saved.")

model = st.session_state.get("model")
if model is None and os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception:
        model = None

st.subheader("Estimate Magnitude")
with st.form("prediction"):
    lat = st.number_input("Latitude", -90.0, 90.0, 17.385)
    lon = st.number_input("Longitude", -180.0, 180.0, 78.4867)
    depth = st.number_input("Depth (km)", 0.0, 800.0, 10.0)
    submit = st.form_submit_button("Estimate")

if submit:
    if model is None:
        st.error("Train the model first.")
    else:
        sample = pd.DataFrame(
            [[lat, lon, depth]], columns=FEATURES
        )
        result = model.predict(sample)[0]
        st.metric("Estimated Magnitude", f"{result:.2f}")
        st.caption("Educational estimate, not an earthquake forecast.")
