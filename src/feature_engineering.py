import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_telco_churn.csv")

print("===== ORIGINAL COLUMNS =====")
print(df.columns.tolist())

# Feature 1: Long-term customer
df["IsLongTermCustomer"] = (df["tenure"] >= 24).astype(int)

# Feature 2: Count additional services
service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

df["TotalServices"] = 0

for column in service_columns:
    df["TotalServices"] += (
        (df[column] != "No") & (df[column] != "No internet service")
    ).astype(int)

print("\n===== NEW FEATURES =====")
print(df[["tenure", "IsLongTermCustomer", "TotalServices", "Churn"]].head(10))

print("\nFeature engineering completed successfully!")

# Save feature-engineered dataset
output_path = "data/feature_engineered_telco_churn.csv"
df.to_csv(output_path, index=False)

print("\nSaved as:", output_path)
