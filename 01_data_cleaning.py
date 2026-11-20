import pandas as pd
import numpy as np

# 1. Load raw dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Fix TotalCharges hidden blank spaces (11 customers with tenure == 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)

# 3. Standardize binary target & SeniorCitizen label
df["Churn_Flag"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

# 4. Feature Engineering for Segmentation & Modeling
# Tenure cohorts
bins = [-1, 12, 24, 48, 72]
labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
df["Tenure_Cohort"] = pd.cut(df["tenure"], bins=bins, labels=labels)

# Active add-on service count (sticky ecosystem indicator)
addon_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
df["Addon_Service_Count"] = (df[addon_cols] == "Yes").sum(axis=1)

# 5. Export cleaned dataset
df.to_csv("telco_churn_cleaned.csv", index=False)

print(f"Cleaned Dataset Shape: {df.shape}")
print(f"Churn Rate: {df['Churn_Flag'].mean()*100:.2f}% ({df['Churn_Flag'].sum():,} churned)")
print(f"Monthly Revenue Lost: ${df.loc[df['Churn_Flag']==1, 'MonthlyCharges'].sum():,.2f}")
print("\nChurn Rate by Tenure Cohort:")
print((df.groupby("Tenure_Cohort", observed=False)["Churn_Flag"].mean() * 100).round(2))