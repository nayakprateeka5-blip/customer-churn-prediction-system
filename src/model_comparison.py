import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/feature_engineered_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. REMOVE CUSTOMER ID
# ============================================================

df = df.drop("customerID", axis=1)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"].map({"Yes": 1, "No": 0})


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns


print("\n===== FEATURES =====")

print("Numerical features:")
print(list(numerical_features))

print("\nCategorical features:")
print(list(categorical_features))


# ============================================================
# 5. CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


print("\n===== TRAIN-TEST SPLIT =====")

print("Total records:", len(X))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))


# ============================================================
# 7. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


print("\n===== MODEL COMPARISON =====")


for name, classifier in models.items():

    print("\nTraining:", name)

    # Create pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability prediction
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, zero_division=0)

    recall = recall_score(y_test, y_pred, zero_division=0)

    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_probability)

    # Store results
    results.append(
        {
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
        }
    )


# ============================================================
# 9. DISPLAY FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n========================================")
print("          FINAL MODEL RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# ============================================================
# 10. FIND BEST MODEL
# ============================================================

best_model = results_df.loc[results_df["F1-Score"].idxmax()]


print("\n========================================")
print("             BEST MODEL")
print("========================================")

print("Best model based on F1-Score:", best_model["Model"])

print("F1-Score:", best_model["F1-Score"])

print("ROC-AUC:", best_model["ROC-AUC"])


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv("reports/model_comparison.csv", index=False)


print("\nModel comparison saved successfully!")

print("File: reports/model_comparison.csv")
