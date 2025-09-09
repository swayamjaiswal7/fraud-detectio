import streamlit as st
import pandas as pd
import joblib

# Load trained XGBoost model
model = joblib.load("fraud_xgb_model.pkl")

st.title("Fraud Detection with XGBoost")


# --- User Inputs ---
amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=100.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=100.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=100.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=100.0)

# Transaction type (0 = TRANSFER, 1 = CASH_OUT)
transaction_type = st.selectbox("Transaction Type", ["TRANSFER (0)", "CASH_OUT (1)"])
type_value = 0 if "TRANSFER" in transaction_type else 1

# --- Build input DataFrame (one step with engineered features) ---
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
    "errorbalanceDest",
    "errorbalanceOrig"
]

user_data = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "type": type_value,
    "errorbalanceDest": oldbalanceDest + amount - newbalanceDest,
    "errorbalanceOrig": newbalanceOrig + amount - oldbalanceOrg
}])[FEATURES]  # enforce order

# --- Predict ---
prediction = model.predict(user_data)[0]

# --- Display results ---
if prediction == 1:
    st.error("🚨 Fraud Detected")
else:
    st.success("✅ Not Fraudulent")
