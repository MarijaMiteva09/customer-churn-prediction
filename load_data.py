import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import joblib
import numpy as np


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

df.drop(columns=['customerID'], inplace=True)

df = pd.get_dummies(df, drop_first=True)


X = df.drop(columns=['Churn'])  
y = df['Churn']  

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🟩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(best_rf, "best_rf_model.pkl")

joblib.dump(scaler, "scaler.pkl")

columns = X.columns.tolist()
joblib.dump(columns, 'columns.pkl')  

X_test_scaled = scaler.transform(X_test)  

background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

explainer = shap.TreeExplainer(best_rf, background_data, feature_perturbation="interventional")


shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):  
    shap_values_selected = shap_values[1]  
else:
    shap_values_selected = shap_values  

shap_values_selected = shap_values_selected[:, :, 1]  

print(f"Shape of shap_values_selected: {shap_values_selected.shape}")
print(f"Shape of X_test_scaled: {X_test_scaled.shape}")

shap.summary_plot(shap_values_selected, X_test_scaled, feature_names=X.columns)

if "TotalCharges" in X.columns:
    shap.dependence_plot("TotalCharges", shap_values_selected, X_test_scaled, feature_names=X.columns)
else:
    print("TotalCharges is not a valid feature in the dataset.")

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values_selected[0], X_test_scaled[0], matplotlib=True)

explainer = LimeTabularExplainer(
    X_train, 
    feature_names=X.columns.tolist(), 
    class_names=['No Churn', 'Churn'], 
    discretize_continuous=True
)

i = 0  
exp = explainer.explain_instance(X_test_scaled[i], best_rf.predict_proba, num_features=5)
exp.show_in_notebook()