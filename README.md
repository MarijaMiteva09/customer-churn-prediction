# Customer Churn Prediction App

This is a machine learning project aimed at predicting customer churn for a telecom company. The project uses a Random Forest classifier to predict whether a customer will churn (leave) based on their data. The app allows users to upload their own CSV data and receive predictions about customer churn. It also includes model explainability using SHAP and LIME for better interpretability.

## Features

- **Customer Churn Prediction**: Users can upload a CSV file with customer data to predict whether the customer will churn (1) or not (0).
- **Model Explainability**: The app leverages SHAP and LIME to provide insights into how the model makes its predictions.
- **Data Preprocessing**: The app handles necessary data preprocessing, including one-hot encoding of categorical variables and scaling the features.
- **Model Accuracy**: The model is trained on a telecom customer dataset and achieves high accuracy in predicting churn.

## Files

- `app.py`: The main Streamlit app script where users can interact with the model.
- `load_data.py`: The script for training the model and saving the necessary files (model, scaler, columns).
- `churn_prediction_model.pkl`: The pickled machine learning model (Random Forest).
- `scaler.pkl`: The pickled scaler used for feature standardization.
- `columns.pkl`: The pickled column names for ensuring proper feature ordering.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for model training (available publicly).

## Requirements

To run this app locally, make sure you have the following dependencies installed:

- `streamlit`
- `pandas`
- `scikit-learn`
- `imblearn`
- `shap`
- `lime`
- `matplotlib`
- `seaborn`
- `joblib`
- `numpy`

You can install all dependencies using pip:

```bash
pip install -r requirements.txt

