import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

if not os.path.exists('churn_prediction_model.pkl'):
    st.error("Model file 'churn_prediction_model.pkl' not found!")
if not os.path.exists('scaler.pkl'):
    st.error("Scaler file 'scaler.pkl' not found!")
if not os.path.exists('columns.pkl'):
    st.error("Columns file 'columns.pkl' not found!")

model = joblib.load('churn_prediction_model.pkl')  
scaler = joblib.load('scaler.pkl')  
columns = joblib.load('columns.pkl')  

def user_input_features():
    st.title("Customer Churn Prediction App")

    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Customer Data")
        st.write(data)
        return data
    else:
        st.warning("Please upload a CSV file to continue.")
        return None

def preprocess_data(data):
    
    data = pd.get_dummies(data, drop_first=True)

    
    missing_cols = set(columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    data = data[columns]
  
    data_scaled = scaler.transform(data)

    return data_scaled

def predict_churn(data):
    if data is not None:
        data_scaled = preprocess_data(data)
        predictions = model.predict(data_scaled)

        st.subheader("Prediction Results")
        churn_count = sum(predictions)  
        non_churn_count = len(predictions) - churn_count  
        
        st.write(f"Total predictions: {len(predictions)}")
        st.write(f"Churn predictions (1): {churn_count}")
        st.write(f"Non-churn predictions (0): {non_churn_count}")

def main():
    data = user_input_features()
    if data is not None:
        with st.spinner('Processing the data and predicting...'):
            predict_churn(data)

if __name__ == "__main__":
    main()
