import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Value Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Predictor")
st.markdown("### Enter Vehicle Specifications")
st.divider()

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("ridge_model.pkl")
    return scaler, model

scaler, model = load_models()

EXPECTED_COLUMNS = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 
    'enginesize', 'horsepower', 'citympg', 'carbody_hardtop', 'carbody_hatchback', 
    'carbody_sedan', 'carbody_wagon', 'drivewheel_fwd', 'drivewheel_rwd', 
    'enginelocation_rear', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 
    'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor', 'cylindernumber_five', 
    'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_three', 
    'cylindernumber_twelve', 'cylindernumber_two'
]

# -----------------------------------------------------------------------------
# INPUT UI
# -----------------------------------------------------------------------------
st.subheader("📏 Vehicle Dimensions")
col1, col2, col3, col4 = st.columns(4)

wheelbase = col1.number_input("Wheelbase", 80.0, 130.0, 98.0)
carlength = col2.number_input("Car Length", 140.0, 210.0, 174.0)
carwidth = col3.number_input("Car Width", 60.0, 80.0, 65.9)
curbweight = col4.number_input("Curb Weight", 1400, 4100, 2500)

st.subheader("⚙️ Engine & Performance")
col5, col6, col7, col8 = st.columns(4)

enginesize = col5.number_input("Engine Size", 60, 350, 120)
horsepower = col6.number_input("Horsepower", 40, 300, 100)
citympg = col7.number_input("City MPG", 10, 60, 25)
symboling = col8.slider("Symboling", -3, 3, 0)

st.subheader("🎨 Vehicle Configuration")
col9, col10, col11, col12, col13 = st.columns(5)

carbody = col9.selectbox("Car Body", 
    ["convertible (default)", "hardtop", "hatchback", "sedan", "wagon"])

drivewheel = col10.selectbox("Drive Wheel", 
    ["4wd (default)", "fwd", "rwd"])

enginetype = col11.selectbox("Engine Type", 
    ["dohc (default)", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])

enginelocation = col12.selectbox("Engine Location", 
    ["front (default)", "rear"])

cylindernumber = col13.selectbox("Cylinders", 
    ["eight (default)", "two", "three", "four", "five", "six", "twelve"])

st.divider()

# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
if st.button("Calculate Estimated Price"):

    input_dict = {col: 0.0 for col in EXPECTED_COLUMNS}

    # numerical
    input_dict['symboling'] = symboling
    input_dict['wheelbase'] = wheelbase
    input_dict['carlength'] = carlength
    input_dict['carwidth'] = carwidth
    input_dict['curbweight'] = curbweight
    input_dict['enginesize'] = enginesize
    input_dict['horsepower'] = horsepower
    input_dict['citympg'] = citympg

    # categorical
    if carbody != "convertible (default)":
        input_dict[f'carbody_{carbody}'] = 1

    if drivewheel != "4wd (default)":
        input_dict[f'drivewheel_{drivewheel}'] = 1

    if enginelocation != "front (default)":
        input_dict[f'enginelocation_{enginelocation}'] = 1

    if enginetype != "dohc (default)":
        input_dict[f'enginetype_{enginetype}'] = 1

    if cylindernumber != "eight (default)":
        input_dict[f'cylindernumber_{cylindernumber}'] = 1

    input_df = pd.DataFrame([input_dict])

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    final_price = max(0, prediction)

    # result card
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#ff4b4b,#ff8c42);
        padding:25px;
        border-radius:12px;
        text-align:center;
        color:white;
        font-size:28px;
        font-weight:bold;
    ">
    Estimated Car Price 💰 <br>
    ${final_price:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.metric("Estimated Vehicle Value", f"${final_price:,.2f}")

    # feature importance
    st.subheader("📊 Feature Impact")
    importance = pd.Series(model.coef_, index=EXPECTED_COLUMNS)
    importance = importance.sort_values(key=abs, ascending=False).head(10)
    st.bar_chart(importance)