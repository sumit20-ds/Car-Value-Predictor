import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.markdown(
"""
<h1 style='text-align:center;'>🚗 Car Price Predictor</h1>
<p style='text-align:center;color:gray;'>
Machine Learning Powered Vehicle Valuation
</p>
""",
unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Value Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, modern look
# st.markdown("""
#     <style>
#     .stButton>button {
#         width: 100%;
#         border-radius: 5px;
#         height: 3em;
#         background-color: #FF4B4B;
#         color: white;
#         font-weight: bold;
#     }
#     .stButton>button:hover {
#         background-color: #FF6B6B;
#         color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)
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

# -----------------------------------------------------------------------------
# 2. MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("ridge_model.pkl")
        return scaler, model
    except FileNotFoundError:
        st.error("⚠️ Could not find `scaler.pkl` or `ridge_model.pkl`. Please ensure they are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the models: {e}")
        st.stop()

scaler, model = load_models()

# The exact 27 columns expected by the scaler (extracted from scaler.pkl)
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
# 3. USER INTERFACE
# -----------------------------------------------------------------------------
# st.title("🚗 Car Value Predictor")
# st.markdown("Enter the vehicle specifications below to get an estimated price based on our Ridge Regression model.")
# st.divider()

# # Organize inputs into tabs for a cleaner UI
# tab1, tab2, tab3 = st.tabs(["📏 Dimensions & Weight", "⚙️ Engine & Performance", "🎨 Style & Configuration"])

# with tab1:
#     st.subheader("Vehicle Dimensions")
#     col1, col2 = st.columns(2)
#     with col1:
#         wheelbase = st.number_input("Wheelbase (inches)", min_value=80.0, max_value=130.0, value=98.0, step=0.1)
#         carlength = st.number_input("Car Length (inches)", min_value=140.0, max_value=210.0, value=174.0, step=0.1)
#     with col2:
#         carwidth = st.number_input("Car Width (inches)", min_value=60.0, max_value=80.0, value=65.9, step=0.1)
#         curbweight = st.number_input("Curb Weight (lbs)", min_value=1400, max_value=4100, value=2500, step=10)

# with tab2:
#     st.subheader("Performance Specs")
#     col3, col4 = st.columns(2)
#     with col3:
#         enginesize = st.number_input("Engine Size (cubic inches)", min_value=60, max_value=350, value=120, step=5)
#         horsepower = st.number_input("Horsepower (HP)", min_value=40, max_value=300, value=100, step=5)
#     with col4:
#         citympg = st.number_input("City MPG", min_value=10, max_value=60, value=25, step=1)
#         symboling = st.slider("Insurance Risk Rating (Symboling)", min_value=-3, max_value=3, value=0, help="-3 is safest, 3 is most risky.")

# with tab3:
#     st.subheader("Categorical Features")
#     col5, col6, col7 = st.columns(3)
#     with col5:
#         carbody = st.selectbox("Car Body Style", ["convertible (default)", "hardtop", "hatchback", "sedan", "wagon"])
#         drivewheel = st.selectbox("Drive Wheel", ["4wd (default)", "fwd", "rwd"])
#     with col6:
#         enginetype = st.selectbox("Engine Type", ["dohc (default)", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])
#         enginelocation = st.selectbox("Engine Location", ["front (default)", "rear"])
#     with col7:
#         cylindernumber = st.selectbox("Number of Cylinders", ["eight (default)", "two", "three", "four", "five", "six", "twelve"])
# -----------------------------------------------------------------------------
# 3. USER INTERFACE
# -----------------------------------------------------------------------------
st.title("🚗 Car Value Predictor")
st.markdown("### Enter Vehicle Specifications")
st.divider()

# ------------------ DIMENSIONS ------------------
st.subheader("📏 Vehicle Dimensions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    wheelbase = st.number_input("Wheelbase", 80.0, 130.0, 98.0)

with col2:
    carlength = st.number_input("Car Length", 140.0, 210.0, 174.0)

with col3:
    carwidth = st.number_input("Car Width", 60.0, 80.0, 65.9)

with col4:
    curbweight = st.number_input("Curb Weight", 1400, 4100, 2500)

# ------------------ ENGINE ------------------
st.subheader("⚙️ Engine & Performance")
col5, col6, col7, col8 = st.columns(4)

with col5:
    enginesize = st.number_input("Engine Size", 60, 350, 120)

with col6:
    horsepower = st.number_input("Horsepower", 40, 300, 100)

with col7:
    citympg = st.number_input("City MPG", 10, 60, 25)

with col8:
    symboling = st.slider("Symboling", -3, 3, 0)

# ------------------ CATEGORICAL ------------------
st.subheader("🎨 Vehicle Configuration")
col9, col10, col11, col12, col13 = st.columns(5)

with col9:
    carbody = st.selectbox("Car Body", 
        ["convertible (default)", "hardtop", "hatchback", "sedan", "wagon"])

with col10:
    drivewheel = st.selectbox("Drive Wheel", 
        ["4wd (default)", "fwd", "rwd"])

with col11:
    enginetype = st.selectbox("Engine Type", 
        ["dohc (default)", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"])

with col12:
    enginelocation = st.selectbox("Engine Location", 
        ["front (default)", "rear"])

with col13:
    cylindernumber = st.selectbox("Cylinders", 
        ["eight (default)", "two", "three", "four", "five", "six", "twelve"])

# -----------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# -----------------------------------------------------------------------------
st.divider()

if st.button("Calculate Estimated Price"):
    with st.spinner("Analyzing vehicle data..."):
        # Initialize a dictionary with all 27 expected columns set to 0.0
        input_dict = {col: 0.0 for col in EXPECTED_COLUMNS}
        
        # 1. Assign Numerical Features
        input_dict['symboling'] = float(symboling)
        input_dict['wheelbase'] = float(wheelbase)
        input_dict['carlength'] = float(carlength)
        input_dict['carwidth'] = float(carwidth)
        input_dict['curbweight'] = float(curbweight)
        input_dict['enginesize'] = float(enginesize)
        input_dict['horsepower'] = float(horsepower)
        input_dict['citympg'] = float(citympg)
        
        # 2. Assign One-Hot Encoded Categorical Features
        # Car Body
        if carbody != "convertible (default)":
            input_dict[f'carbody_{carbody}'] = 1.0
            
        # Drive Wheel
        if drivewheel != "4wd (default)":
            input_dict[f'drivewheel_{drivewheel}'] = 1.0
            
        # Engine Location
        if enginelocation != "front (default)":
            input_dict[f'enginelocation_{enginelocation}'] = 1.0
            
        # Engine Type
        if enginetype != "dohc (default)":
            input_dict[f'enginetype_{enginetype}'] = 1.0
            
        # Cylinders
        if cylindernumber != "eight (default)":
            input_dict[f'cylindernumber_{cylindernumber}'] = 1.0

        # 3. Create DataFrame and Predict
        # Using a list containing the dictionary ensures it creates a 1-row DataFrame
        input_df = pd.DataFrame([input_dict])
        
        try:
            # Scale the features
            scaled_features = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(scaled_features)[0]
            
            # Ensure price doesn't output as a negative number by accident
            final_price = max(0, prediction)
            
            st.success("Prediction Complete!")
            st.metric(label="Estimated Vehicle Value", value=f"${final_price:,.2f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Check your feature inputs or model compatibility.")

st.subheader("📊 Feature Impact")

importance = pd.Series(model.coef_, index=EXPECTED_COLUMNS)
importance = importance.sort_values(key=abs, ascending=False).head(10)

st.bar_chart(importance)