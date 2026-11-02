import os
import requests
import xml.etree.ElementTree as ET
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

def parse_quakeml(file_content):
    """Parses a QuakeML XML file and returns a pandas DataFrame."""
    root = ET.fromstring(file_content)
    
    # Define common namespaces used in QuakeML
    namespaces = {
        'q': 'http://quakeml.org',
        'anss': 'http://anss.org',
        'catalog': 'http://anss.org'
    }
    
    rows = []
    # Find all event elements (handling default namespace with wildcard or explicit check)
    for event in root.findall('.//{http://quakeml.org}event'):
        try:
            # Extract origin parameters
            origin = event.find('{http://quakeml.org}origin')
            lat_val = origin.find('{http://quakeml.org}latitude/{http://quakeml.org}value')
            lon_val = origin.find('{http://quakeml.org}longitude/{http://quakeml.org}value')
            depth_val = origin.find('{http://quakeml.org}depth/{http://quakeml.org}value')
            
            # Extract magnitude parameters
            magnitude = event.find('{http://quakeml.org}magnitude')
            mag_val = magnitude.find('{http://quakeml.org}mag/{http://quakeml.org}value')
            
            if lat_val is not None and lon_val is not None and mag_val is not None:
                # QuakeML depths are usually in meters, convert to km if needed
                depth_km = float(depth_val.text) / 1000.0 if depth_val is not None else 0.0
                
                rows.append({
                    "latitude": float(lat_val.text),
                    "longitude": float(lon_val.text),
                    "depth": depth_km,
                    "magnitude": float(mag_val.text)
                })
        except Exception:
            continue
            
    return pd.DataFrame(rows)

def load_data():
    uploaded = st.sidebar.file_uploader("Upload earthquake file", type=["csv", "xml"])
    if uploaded:
        if uploaded.name.endswith('.xml'):
            return parse_quakeml(uploaded.read())
        else:
            return pd.read_csv(uploaded)

    if os.path.exists("earthquakes.csv"):
        return pd.read_csv("earthquakes.csv")

    st.sidebar.info("No file found. Download recent data from USGS.")
    if st.sidebar.button("Download USGS data"):
        response = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": "2020-01-01",
                "minmagnitude": 2.5,
                "limit": 2000
            },
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
    st.info("👋 Welcome! Please upload a data file (.csv or .xml) in the sidebar or click 'Download USGS data' to get started.")
    st.stop()


df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    "lat": "latitude", "lon": "longitude",
    "mag": "magnitude", "depth_km": "depth"
})

required = FEATURES + ["magnitude"]
if not all(c in df.columns for c in required):
    st.error("Data must contain latitude, longitude, depth, and magnitude.")
    st.stop()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required)
df = df[df["latitude"].between(-90, 90)]
df = df[df["longitude"].between(-180, 180)]
df = df[df["magnitude"].between(-2, 10)]

if len(df) < 20:
    st.error(f"At least 20 valid records are required. Found {len(df)}.")
    st.stop()

st.success(f"Loaded {len(df)} earthquake records.")
st.dataframe(df.head(20), use_container_width=True)

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
