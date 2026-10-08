import pandas as pd

# Load dataset
file_path = "data/telco_churn.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# 1. Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 2. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 3. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4. Check missing values again
print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# 5. Remove rows with missing values
df = df.dropna()

print("\nDataset shape after removing missing values:")
print(df.shape)

# 6. Save cleaned dataset
output_path = "data/cleaned_telco_churn.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved as:", output_path)
