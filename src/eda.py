import pandas as pd

df = pd.read_csv("data/cleaned_telco_churn.csv")

print("===== CHURN PERCENTAGE =====")

churn_percentage = df["Churn"].value_counts(normalize=True).mul(100).round(2)

print(churn_percentage)

print("\n===== CLASS IMBALANCE CHECK =====")

for category, percentage in churn_percentage.items():
    print(f"{category}: {percentage}%")
