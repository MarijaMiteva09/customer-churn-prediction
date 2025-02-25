import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive mode
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

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ✅ Convert 'TotalCharges' to numeric (handle missing values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# ✅ Convert 'Churn' to binary (Yes=1, No=0)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# ✅ Drop customerID (not useful for predictions)
df.drop(columns=['customerID'], inplace=True)

# ✅ One-Hot Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# ------------------------
# 🔥 Model Training
# ------------------------

# Split dataset into training and testing sets
X = df.drop(columns=['Churn'])  # Features
y = df['Churn']  # Target

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Hyperparameter Tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Train the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# ✅ Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🟩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
import joblib
joblib.dump(best_rf, "churn_prediction_model.pkl")
