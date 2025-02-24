import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive mode
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display basic information about the dataset
print("Dataset Overview:\n")
print(df.head())
print("\nSummary:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())

# ✅ Convert 'TotalCharges' to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# ✅ Convert 'Churn' to binary (Yes=1, No=0)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# ------------------------
# ✅ Data Visualization
# ------------------------

# 1️⃣ Boxplot: MonthlyCharges vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.xlabel('Churn (1 = Yes, 0 = No)')
plt.ylabel('Monthly Charges')
plt.savefig("monthly_charges_vs_churn.png")  # Save plot
plt.close()

# 2️⃣ Correlation Heatmap
corr_matrix = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig("correlation_heatmap.png")  # Save plot
plt.close()

# 3️⃣ Countplot: Gender vs Churn
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Churn by Gender')
plt.savefig("churn_by_gender.png")  # Save plot
plt.close()

# 4️⃣ Countplot: Partner vs Churn
plt.figure(figsize=(8, 6))
sns.countplot(x='Partner', hue='Churn', data=df)
plt.title('Churn by Partner')
plt.savefig("churn_by_partner.png")  # Save plot
plt.close()

# ------------------------
# ✅ Hypothesis Testing
# ------------------------

# 5️⃣ T-test: MonthlyCharges between churned and non-churned customers
churned = df[df['Churn'] == 1]['MonthlyCharges']
non_churned = df[df['Churn'] == 0]['MonthlyCharges']

t_stat, p_val = stats.ttest_ind(churned, non_churned)
print("\nT-test for Monthly Charges & Churn:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")

if p_val < 0.05:
    print("✅ Significant difference: Monthly Charges affects Churn.")
else:
    print("❌ No significant difference: Monthly Charges does NOT affect Churn.")

# 6️⃣ Chi-Square Test: Relationship between Partner and Churn
contingency = pd.crosstab(df['Partner'], df['Churn'])
chi2, p_val, _, _ = stats.chi2_contingency(contingency)
print("\nChi-Square Test for Partner & Churn:")
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_val}")

if p_val < 0.05:
    print("✅ Significant relationship: Having a Partner affects Churn.")
else:
    print("❌ No significant relationship: Partner status does NOT affect Churn.")
