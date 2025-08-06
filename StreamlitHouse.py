# house_price_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Prepare model
x = data.drop('PRICE', axis=1)
y = data['PRICE']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(xtrain, ytrain)

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="California House Price Predictor", layout="wide")

# Custom style
st.markdown("""
    <style>
    .main {
        background-color: #f0f7ff;
    }
    .stApp {
        background: linear-gradient(to bottom right, #e6f7ff, #ffffff);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 California House Price Prediction App")
st.markdown("Enter the house details below to predict the price.")

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    MedInc = st.slider('Median Income (10k USD)', 0.5, 15.0, 3.0)
    HouseAge = st.slider('House Age (years)', 1, 50, 20)

with col2:
    AveRooms = st.slider('Average Rooms', 1.0, 10.0, 5.0)
    AveBedrms = st.slider('Average Bedrooms', 0.5, 5.0, 1.0)

with col3:
    Population = st.slider('Population in Block', 100, 5000, 1500)
    AveOccup = st.slider('Average Occupancy', 1.0, 10.0, 3.0)

Latitude = st.slider('Latitude', 32.0, 42.0, 36.0)
Longitude = st.slider('Longitude', -124.0, -114.0, -119.0)

# Predict button
if st.button("Predict Price 💰"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)[0]
    st.success(f"🏠 Predicted House Price: **${prediction * 100000:,.2f}**")

# Show model info
with st.expander("📈 View Model Info & Coefficients"):
    coef_df = pd.DataFrame(model.coef_, index=housing.feature_names, columns=["Coefficient"])
    st.dataframe(coef_df.style.highlight_max(axis=0, color='lightgreen'))
