import pandas as pd

# Sample data for customer churn prediction
data = {
    "gender": ["Female", "Male", "Female", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0, 0, 1],
    "Partner": ["Yes", "Yes", "No", "No", "Yes"],
    "Dependents": ["No", "No", "No", "Yes", "Yes"],
    "tenure": [1, 34, 2, 45, 12],
    "PhoneService": ["Yes", "Yes", "No", "Yes", "No"],
    "MultipleLines": ["No phone service", "Yes", "No phone service", "Yes", "No"],
    "InternetService": ["DSL", "Fiber optic", "Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No", "Yes", "No"],
    "OnlineBackup": ["Yes", "Yes", "Yes", "No", "No"],
    "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
    "TechSupport": ["No", "Yes", "Yes", "Yes", "No"],
    "StreamingTV": ["No", "Yes", "No", "Yes", "No"],
    "StreamingMovies": ["No", "Yes", "Yes", "Yes", "No"],
    "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month"],
    "PaperlessBilling": ["Yes", "Yes", "No", "Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"],
    "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
    "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 711.45],
    "Churn": [0, 1, 0, 1, 0]  # 0: No churn, 1: Churn
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sample_customer_data.csv", index=False)
