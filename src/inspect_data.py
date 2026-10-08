import pandas as pd

# Load the dataset
file_path = "data/telco_churn.csv"

df = pd.read_csv(file_path)

# Show first 5 rows
print("===== FIRST 5 ROWS =====")
print(df.head())

# Show number of rows and columns
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Show column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# Show data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Check duplicate rows
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# Check target variable
print("\n===== CHURN DISTRIBUTION =====")
print(df["Churn"].value_counts())
