# -Earthquake-Prediction-Using-ML
# 🌍 Earthquake Prediction Using ML

An interactive Streamlit web application that uses Machine Learning (Random Forest Regression) to estimate earthquake magnitudes based on geographical coordinates and depth features. 

⚠️ **Disclaimer:** This project is for educational use only. Machine learning models cannot reliably predict future earthquakes.

---

## 🚀 Features
- **Flexible Data Ingestion:** Supports manual file uploads for both CSV and QuakeML (.xml) formats.
- **Automated Fallback:** Fetches live real-time global earthquake feeds directly from the USGS API.
- **Data Visualization:** Built-in interactive map plotting and magnitude distribution histograms.
- **Machine Learning Pipeline:** Trains a Random Forest Regressor on 80% of the dataset and evaluates performance metrics (MAE and RMSE).
- **Interactive Predictor:** Allows custom testing of latitude, longitude, and depth to instantly estimate magnitude values.

---

## 🛠️ Installation & Setup

1. **Clone or navigate to your project directory:**
   ```bash
   cd path/to/your/earthquake-project
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run akkhil.py
   ```

---

## 📁 File Structure
```text
├── akkhil.py                 # Main Streamlit application source code
├── requirements.txt          # Python dependencies list
├── README.md                 # Project documentation
└── earthquake_model.joblib   # Trained model file (generated after training)
```

---

## 📊 Data Source
- **Primary Source:** Data is requested dynamically from the United States Geological Survey (USGS) API.
- **Manual Uploads:** If your network blocks the direct USGS endpoint, you can manually download and feed data from the [USGS Earthquake Hazards Program](https://usgs.gov).
