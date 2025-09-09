import streamlit as st
import pandas as pd
import joblib

# --- Load trained XGBoost model ---
model = joblib.load("fraud_xgb_model.pkl")

st.title("Fraud Detection App")

# --- User Inputs ---
amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=100.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=100.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=100.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=100.0)

# Encode transaction type: TRANSFER=0, CASH_OUT=1
transaction_type = st.selectbox("Transaction Type", ["TRANSFER (0)", "CASH_OUT (1)"])
type_value = 0 if "TRANSFER" in transaction_type else 1

# --- Define feature order (exactly as model expects) ---
FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorbalanceDest",
    "errorbalanceOrig"
]

# --- Build input DataFrame ---
user_data = pd.DataFrame([{
    "type": type_value,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "errorbalanceDest": oldbalanceDest + amount - newbalanceDest,
    "errorbalanceOrig": newbalanceOrig + amount - oldbalanceOrg
}])[FEATURES]  # enforce names + order

# --- Debug (optional) ---
# st.write("Model expects:", model.get_booster().feature_names)
# st.write("User data columns:", list(user_data.columns))

# --- Prediction ---
if st.button("Predict Fraud"):
    prediction = model.predict(user_data)[0]

    if prediction == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ The transaction is legit", prediction)
