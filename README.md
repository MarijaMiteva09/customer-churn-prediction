Customer Churn Prediction App
This is a machine learning project aimed at predicting customer churn for a telecom company. The project uses a Random Forest classifier to predict whether a customer will churn (leave) based on their data. The app allows users to upload their own CSV data and receive predictions about customer churn. It also includes model explainability using SHAP and LIME for better interpretability.

Features
Customer Churn Prediction: Users can upload a CSV file with customer data to predict whether the customer will churn (1) or not (0).
Model Explainability: The app leverages SHAP and LIME to provide insights into how the model makes its predictions.
Data Preprocessing: The app handles necessary data preprocessing, including one-hot encoding of categorical variables and scaling the features.
Model Accuracy: The model is trained on a telecom customer dataset and achieves high accuracy in predicting churn.
Files
app.py: The main Streamlit app script where users can interact with the model.
load_data.py: The script for training the model and saving the necessary files (model, scaler, columns).
churn_prediction_model.pkl: The pickled machine learning model (Random Forest).
scaler.pkl: The pickled scaler used for feature standardization.
columns.pkl: The pickled column names for ensuring proper feature ordering.
WA_Fn-UseC_-Telco-Customer-Churn.csv: The dataset used for model training (available publicly).
Requirements
To run this app locally, make sure you have the following dependencies installed:

streamlit
pandas
scikit-learn
imblearn
shap
lime
matplotlib
seaborn
joblib
numpy
You can install all dependencies using pip:

bash
Copy
Edit
pip install -r requirements.txt
Here is a sample requirements.txt file:

ini
Copy
Edit
streamlit==1.7.0
pandas==1.3.3
scikit-learn==0.24.2
imblearn==0.0
shap==0.39.0
lime==0.2.0.1
matplotlib==3.4.3
seaborn==0.11.2
joblib==1.0.1
numpy==1.21.2
Usage
Train the Model:

Run load_data.py to preprocess the data, train the model, and save the model, scaler, and column files.
This will generate the following files:
churn_prediction_model.pkl
scaler.pkl
columns.pkl
Run the Streamlit App:

After training the model and saving the necessary files, run the Streamlit app by using the following command:
bash
Copy
Edit
streamlit run app.py
Upload a CSV File:

The app will allow you to upload a CSV file containing customer data. The CSV file should have the same structure as the original dataset.
The app will preprocess the data, make predictions, and display the results.
View Prediction Results:

The prediction results will show which customers are predicted to churn (1) and which are not (0).
You can also view the model explainability plots using SHAP and LIME.
Model Evaluation
The model is trained using Random Forest and optimized through hyperparameter tuning using GridSearchCV. It achieves an accuracy of XX% on the test dataset, which makes it effective for predicting churn.

Example of the output:
Total Predictions: 1000
Churn Predictions (1): 300
Non-Churn Predictions (0): 700
Model Explainability
This project incorporates explainability tools like SHAP and LIME to help users understand why the model is making specific predictions.

SHAP (Shapley Additive Explanations): This method provides feature-level importance for individual predictions. The app displays a summary plot of SHAP values to visualize which features contributed the most to the model's decision.
LIME (Local Interpretable Model-agnostic Explanations): LIME is used to explain individual predictions by approximating the model locally with simpler models.
Troubleshooting
Missing Files: Ensure that the following files are present before running the app:

churn_prediction_model.pkl
scaler.pkl
columns.pkl
Data Format: The CSV file uploaded by the user should have the same structure as the original dataset used for training. Ensure that the file includes all necessary columns and follows the same format.

License
This project is open-source and available under the MIT License.
