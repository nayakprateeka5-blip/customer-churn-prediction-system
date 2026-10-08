import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/feature_engineered_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. REMOVE CUSTOMER ID
# ============================================================

df = df.drop("customerID", axis=1)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"].map({"Yes": 1, "No": 0})


# ============================================================
# 4. IDENTIFY FEATURES
# ============================================================

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns


# ============================================================
# 5. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ============================================================
# 6. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 8. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": ["balanced", None],
    "classifier__solver": ["liblinear", "lbfgs"],
}


# ============================================================
# 9. GRID SEARCH
# ============================================================

print("\n========================================")
print("      HYPERPARAMETER OPTIMIZATION")
print("========================================")

grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
)


print("\nStarting GridSearchCV...")

grid_search.fit(X_train, y_train)


# ============================================================
# 10. BEST PARAMETERS
# ============================================================

print("\n========================================")
print("          BEST PARAMETERS")
print("========================================")

print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:", round(grid_search.best_score_, 4))


# ============================================================
# 11. TEST SET EVALUATION
# ============================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

y_probability = best_model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_probability)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("       OPTIMIZED MODEL PERFORMANCE")
print("========================================")

print("Accuracy :", round(accuracy, 4))

print("Precision:", round(precision, 4))

print("Recall   :", round(recall, 4))

print("F1-Score :", round(f1, 4))

print("ROC-AUC  :", round(roc_auc, 4))


print("\nHyperparameter optimization completed!")
