import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ========================================
# 1. LOAD DATA
# ========================================

df = pd.read_csv("data/feature_engineered_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ========================================
# 2. PREPARE DATA
# ========================================

df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)

y = df["Churn"].map({"Yes": 1, "No": 0})


# ========================================
# 3. IDENTIFY FEATURES
# ========================================

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns


# ========================================
# 4. PREPROCESSING
# ========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ========================================
# 5. FINAL MODEL
# ========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=1, class_weight="balanced", solver="liblinear", max_iter=2000
            ),
        ),
    ]
)


# ========================================
# 6. TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ========================================
# 7. TRAIN MODEL
# ========================================

print("\n========================================")
print("          TRAINING FINAL MODEL")
print("========================================")

model.fit(X_train, y_train)

print("Final model trained successfully!")


# ========================================
# 8. MAKE PREDICTIONS
# ========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ========================================
# 9. CALCULATE METRICS
# ========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_probability)


# ========================================
# 10. DISPLAY RESULTS
# ========================================

print("\n========================================")
print("          FINAL MODEL RESULTS")
print("========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ========================================
# 11. CLASSIFICATION REPORT
# ========================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0
    )
)


# ========================================
# 12. CONFUSION MATRIX
# ========================================

cm = confusion_matrix(y_test, y_pred)

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)

plt.title("Confusion Matrix - Customer Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("reports/final_confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully!")


# ========================================
# 13. ROC CURVE
# ========================================

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Customer Churn Prediction")

plt.legend()

plt.tight_layout()

plt.savefig("reports/roc_curve.png")

plt.show()

print("ROC curve saved successfully!")


# ========================================
# 14. SAVE FINAL MODEL
# ========================================

import joblib

joblib.dump(model, "models/churn_model.joblib")

print("\nFinal model saved successfully!")

print("File: models/churn_model.joblib")


# ========================================
# COMPLETE
# ========================================

print("\n========================================")
print("      MODEL EVALUATION COMPLETED")
print("========================================")
