
import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Earthquake Prediction",
    layout="wide"
)

st.title("🌍 Earthquake Prediction Using ML")

st.warning(
    "Educational use only. This application cannot reliably predict future earthquakes."
)

FEATURES = ["latitude", "longitude", "depth"]
MODEL_FILE = "earthquake_model.joblib"

# ---------------- XML PARSER ----------------

def parse_quakeml(file_content):

    root = ET.fromstring(file_content)
    rows = []

    ns = {
        "q": "http://quakeml.org/xmlns/quakeml/1.2",
        "b": "http://quakeml.org/xmlns/bed/1.2"
    }

    for event in root.findall(".//b:event", ns):

        try:
            origin = event.find("b:origin", ns)
            magnitude = event.find("b:magnitude", ns)

            if origin is None or magnitude is None:
                continue

            lat = origin.find("b:latitude/b:value", ns)
            lon = origin.find("b:longitude/b:value", ns)
            depth = origin.find("b:depth/b:value", ns)
            mag = magnitude.find("b:mag/b:value", ns)

            if lat is None or lon is None or mag is None:
                continue

            rows.append({
                "latitude": float(lat.text),
                "longitude": float(lon.text),
                "depth": float(depth.text) / 1000 if depth is not None else 0,
                "magnitude": float(mag.text)
            })

        except (ValueError, AttributeError, TypeError):
            continue

    return pd.DataFrame(rows)


# ---------------- DOWNLOAD USGS DATA ----------------

def download_usgs_data():

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": "2020-01-01",
        "minmagnitude": 2.5,
        "limit": 2000,
        "orderby": "time"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    rows = []

    for event in response.json().get("features", []):

        properties = event.get("properties") or {}
        geometry = event.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])

        if len(coordinates) >= 3 and properties.get("mag") is not None:

            rows.append({
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "depth": coordinates[2],
                "magnitude": properties["mag"],
                "time": pd.to_datetime(
                    properties.get("time"),
                    unit="ms",
                    utc=True
                )
            })

    return pd.DataFrame(rows)


# ---------------- LOAD DATA ----------------

def load_data():

    uploaded = st.sidebar.file_uploader(
        "Upload Earthquake CSV or XML",
        type=["csv", "xml"]
    )

    if uploaded is not None:

        if uploaded.name.lower().endswith(".xml"):
            return parse_quakeml(uploaded.read())

        return pd.read_csv(uploaded)

    if os.path.exists("earthquakes.csv"):
        return pd.read_csv("earthquakes.csv")

    st.sidebar.info("Upload a file or download USGS records.")

    if st.sidebar.button("🌍 Download USGS Data"):

        with st.spinner("Downloading earthquake records..."):
            return download_usgs_data()

    return None


# ---------------- COUNTRY IDENTIFICATION ----------------

def identify_countries(df):

    df = df.copy()

    coordinates = list(
        zip(df["latitude"], df["longitude"])
    )

    results = rg.search(coordinates, mode=1)

    df["country_code"] = [
        result["cc"] for result in results
    ]

    country_names = {}

    for country in pycountry.countries:
        country_names[country.alpha_2] = country.name

    df["country"] = [
        country_names.get(code, "Unknown")
        for code in df["country_code"]
    ]

    return df


# ---------------- MAIN DATA PROCESSING ----------------

try:

    df = load_data()

except Exception as e:

    st.error(f"Data loading failed: {e}")
    st.stop()


if df is None:

    st.info(
        "Please upload a CSV/XML file or click Download USGS Data."
    )
    st.stop()


# Standardize column names

df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    "lat": "latitude",
    "lon": "longitude",
    "mag": "magnitude",
    "depth_km": "depth",
    "time_utc": "time",
    "date": "time"
})


# ---------------- VALIDATE DATA ----------------

required = FEATURES + ["magnitude"]

if not all(column in df.columns for column in required):

    st.error(
        "Dataset must contain latitude, longitude, depth and magnitude."
    )
    st.stop()


for column in required:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(subset=required)

df = df[
    df["latitude"].between(-90, 90)
]

df = df[
    df["longitude"].between(-180, 180)
]

df = df[
    df["depth"].between(0, 800)
]

df = df[
    df["magnitude"].between(-2, 10)
]


if len(df) < 20:

    st.error(
        f"At least 20 valid earthquake records are required. Found {len(df)}."
    )
    st.stop()


# ---------------- COUNTRY IDENTIFICATION ----------------

try:

    df = identify_countries(df)

except Exception as e:

    st.error(f"Country identification failed: {e}")
    st.stop()


st.success(
    f"Successfully loaded {len(df)} earthquake records."
)


# ---------------- DASHBOARD SUMMARY ----------------

st.subheader("📊 Worldwide Earthquake Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Earthquakes",
    len(df)
)

col2.metric(
    "Countries Identified",
    df["country"].nunique()
)

col3.metric(
    "Average Magnitude",
    round(df["magnitude"].mean(), 2)
)


# ---------------- COUNTRY BUTTONS ----------------

st.subheader("🌍 Select Country")

countries = sorted(
    df["country"].dropna().unique()
)

if "selected_country" not in st.session_state:

    st.session_state.selected_country = "All Countries"


# All Countries button

if st.button(
    "🌎 All Countries",
    use_container_width=True
):

    st.session_state.selected_country = "All Countries"


# Separate country buttons

for start in range(0, len(countries), 4):

    cols = st.columns(4)

    for index, country in enumerate(
        countries[start:start + 4]
    ):

        with cols[index]:

            if st.button(
                country,
                key=f"country_{start}_{index}",
                use_container_width=True
            ):

                st.session_state.selected_country = country


# ---------------- FILTER RECORDS ----------------

if st.session_state.selected_country == "All Countries":

    display_df = df.copy()

else:

    display_df = df[
        df["country"] == st.session_state.selected_country
    ].copy()


st.subheader(
    f"🌍 Earthquake Records - {st.session_state.selected_country}"
)

st.metric(
    "Selected Earthquake Records",
    len(display_df)
)


# ---------------- DISPLAY TABLE ----------------

if display_df.empty:

    st.warning("No records available.")

else:

    visible_columns = [
        "country",
        "latitude",
        "longitude",
        "depth",
        "magnitude"
    ]

    if "time" in display_df.columns:
        visible_columns.insert(1, "time")

    st.dataframe(
        display_df[visible_columns],
        use_container_width=True
    )


    # ---------------- EARTHQUAKE MAP ----------------

    st.subheader("📍 Earthquake Location Map")

    st.map(
        display_df[
            ["latitude", "longitude"]
        ].rename(
            columns={
                "latitude": "lat",
                "longitude": "lon"
            }
        )
    )


    # ---------------- MAGNITUDE GRAPH ----------------

    st.subheader("📊 Magnitude Analysis")

    fig, ax = plt.subplots(figsize=(10, 4))

    display_df["magnitude"].hist(
        bins=20,
        ax=ax
    )

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of Earthquakes")

    ax.set_title(
        f"Magnitude Analysis - {st.session_state.selected_country}"
    )

    st.pyplot(fig)

    plt.close(fig)


    # ---------------- CSV DOWNLOAD ----------------

    csv_data = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Selected Country CSV",
        data=csv_data,
        file_name="earthquake_records.csv",
        mime="text/csv"
    )


# ---------------- MACHINE LEARNING ----------------

st.divider()

st.subheader("🤖 Train Machine Learning Model")

if st.button("Train Model"):

    # Randomize records before splitting
    shuffled = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    split = int(len(shuffled) * 0.8)

    train = shuffled.iloc[:split]
    test = shuffled.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        train[FEATURES],
        train["magnitude"]
    )

    predictions = model.predict(
        test[FEATURES]
    )

    mae = mean_absolute_error(
        test["magnitude"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["magnitude"],
            predictions
        )
    )

    col1, col2 = st.columns(2)

    col1.metric("MAE", f"{mae:.3f}")

    col2.metric("RMSE", f"{rmse:.3f}")

    joblib.dump(
        model,
        MODEL_FILE
    )

    st.session_state["model"] = model

    st.success("Model trained and saved successfully!")


# ---------------- LOAD SAVED MODEL ----------------

model = st.session_state.get("model")

if model is None and os.path.exists(MODEL_FILE):

    try:

        model = joblib.load(MODEL_FILE)

    except Exception:

        model = None


# ---------------- MAGNITUDE ESTIMATION ----------------

st.divider()

st.subheader("🔍 Estimate Earthquake Magnitude")

with st.form("prediction_form"):

    lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=17.385
    )

    lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=78.4867
    )

    depth = st.number_input(
        "Depth (km)",
        min_value=0.0,
        max_value=800.0,
        value=10.0
    )

    submit = st.form_submit_button(
        "Estimate Magnitude"
    )


if submit:

    if model is None:

        st.error("Please train the model first.")

    else:

        sample = pd.DataFrame(
            [[lat, lon, depth]],
            columns=FEATURES
        )

        result = model.predict(sample)[0]

        st.metric(
            "Estimated Magnitude",
            f"{result:.2f}"
        )

        st.caption(
            "Educational ML estimate only. "
            "This is not a reliable earthquake forecast."
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Earthquake Prediction Using Machine Learning | "
    "Data source: USGS Earthquake Catalog | "
    "Educational Diploma CSE Project"
)
