import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

df = pd.read_csv("telco_churn_cleaned.csv")

# --- 1. Statistical Hypothesis Tests ---
ct = pd.crosstab(df["Contract"], df["Churn_Flag"])
chi2, p_chi2, _, _ = stats.chi2_contingency(ct)

churn_charges = df.loc[df["Churn_Flag"] == 1, "MonthlyCharges"]
stay_charges = df.loc[df["Churn_Flag"] == 0, "MonthlyCharges"]
u_stat, p_mw = stats.mannwhitneyu(churn_charges, stay_charges)

print("=== STATISTICAL VALIDATION ===")
print(f"Contract vs Churn Chi-Square: chi2={chi2:.2f}, p={p_chi2:.2e}")
print(f"MonthlyCharges Median — Churned: ${churn_charges.median():.2f} | Retained: ${stay_charges.median():.2f} (Mann-Whitney p={p_mw:.2e})")

# --- 2. Churn Risk Scoring Model (Logistic Regression) ---
# Note: Use raw SeniorCitizen (0/1) if mapped to Yes/No in 01_data_cleaning.py
senior_col = "SeniorCitizen"
features = [
    senior_col, "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Addon_Service_Count"
]

X = df[features]
y = df["Churn_Flag"]

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Addon_Service_Count"]
cat_cols = [c for c in features if c not in num_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)

test_probs = model.predict_proba(X_test)[:, 1]
print("\n=== PREDICTIVE RISK MODEL ===")
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.3f} | Test Accuracy: {accuracy_score(y_test, (test_probs >= 0.5).astype(int)):.3f}")

df["Churn_Risk_Score"] = model.predict_proba(X)[:, 1]
active_high_risk = df[(df["Churn_Flag"] == 0) & (df["Churn_Risk_Score"] >= 0.50)]
print(f"Active High-Risk Customers (Score >= 0.50): {len(active_high_risk):,} | Monthly Revenue at Risk: ${active_high_risk['MonthlyCharges'].sum():,.2f}")

df.to_csv("telco_churn_scored.csv", index=False)

# --- 3. 4-Panel Executive Visual Summary ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Telco Customer Churn Drivers, Risk Scoring & Revenue Impact", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Churn Rate by Contract Type
contract_churn = (df.groupby("Contract")["Churn_Flag"].mean() * 100).sort_values(ascending=False)
sns.barplot(x=contract_churn.index, y=contract_churn.values, ax=axes[0, 0], palette=["#E50914", "#F59E0B", "#3B82F6"])
axes[0, 0].set_title("Churn Rate by Contract Type (%)", fontweight="bold")
axes[0, 0].set_ylabel("Churn Rate (%)")
axes[0, 0].set_xlabel("")
for i, v in enumerate(contract_churn.values):
    axes[0, 0].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold")

# Panel 2: Churn Rate by Tenure Cohort
cohort_order = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
cohort_churn = (df.groupby("Tenure_Cohort", observed=False)["Churn_Flag"].mean() * 100).reindex(cohort_order)
sns.barplot(x=cohort_churn.index, y=cohort_churn.values, ax=axes[0, 1], color="#1E293B")
axes[0, 1].set_title("Churn Rate by Tenure Cohort (%)", fontweight="bold")
axes[0, 1].set_ylabel("Churn Rate (%)")
axes[0, 1].set_xlabel("")
for i, v in enumerate(cohort_churn.values):
    axes[0, 1].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold")

# Panel 3: Monthly Charges Distribution by Churn Status
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0], palette={"No": "#3B82F6", "Yes": "#E50914"})
axes[1, 0].set_title("Monthly Charges by Churn Status ($)", fontweight="bold")
axes[1, 0].set_xlabel("Churned")
axes[1, 0].set_ylabel("Monthly Charges ($)")

# Panel 4: Predicted Churn Risk Score Distribution (Active Customers)
active_df = df[df["Churn_Flag"] == 0]
sns.histplot(active_df["Churn_Risk_Score"], bins=20, kde=True, ax=axes[1, 1], color="#0F766E")
axes[1, 1].axvline(0.50, color="#E50914", linestyle="--", linewidth=2, label="High-Risk Threshold (0.50)")
axes[1, 1].set_title("Predicted Churn Risk Score — Active Customers", fontweight="bold")
axes[1, 1].set_xlabel("Predicted Churn Probability")
axes[1, 1].set_ylabel("Active Customers")
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("telco_churn_executive_summary.png", dpi=300)
print("Saved chart: telco_churn_executive_summary.png")