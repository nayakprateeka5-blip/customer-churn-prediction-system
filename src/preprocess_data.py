import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/cleaned_telco_churn.csv")

# Remove customer ID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})


# Numerical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

# Categorical columns
categorical_features = X.select_dtypes(include=["object"]).columns


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


print("===== TRAIN-TEST SPLIT =====")
print("Total records:", len(X))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining percentage:", round(len(X_train) / len(X) * 100, 2), "%")

print("Testing percentage:", round(len(X_test) / len(X) * 100, 2), "%")

print("\nPreprocessing and train-test split completed successfully!")
