import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)

# Load the dataset
df = pd.read_csv("pcos_dataset.csv")

# Define features and target
features = ["Age", "BMI", "Menstrual_Irregularity", "Testosterone_Level(ng/dL)", "Antral_Follicle_Count"]
target = "PCOS_Diagnosis"

# Sampling
df_sampled = df.sample(n=800, random_state=42)
X = df_sampled[features]
y = df_sampled[target]

# Normalization
scaler = MinMaxScaler()
X[features] = scaler.fit_transform(X[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define parameter grid for Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'class_weight': [None, {0: 1, 1: 2}, {0: 1, 1: 3}],
    'probability': [True]  # Required for ROC curve
}

# Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and estimator
print("\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Obese", "Obese"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Misclassified examples
df_misclassified = X_test.copy()
df_misclassified["Actual"] = y_test.values
df_misclassified["Predicted"] = y_pred
df_misclassified = df_misclassified[df_misclassified["Actual"] != df_misclassified["Predicted"]]

print("\nMisclassified Examples:")
print(df_misclassified.head())

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PCOS Obesity Classification (Grid Search SVM)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
