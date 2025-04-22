import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "dataset 3 kelas.xlsx"  # Change to your correct path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_imputed.drop(columns=["Academic_Performance"])
y = df_imputed["Academic_Performance"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale data (for SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "SVM": SVC(kernel='linear', random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# Train models & evaluate
results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"Training {name}...")
    
    if name in ["SVM", "LogisticRegression"]:
        model.fit(X_train_scaled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
        "confusion_matrix": conf_matrix
    }
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, "best_model.pkl")
print(f"Best model saved: {best_model_name} with accuracy {best_accuracy:.4f}")

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{name} - Confusion Matrix")
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")

plt.tight_layout()
plt.show()

# Plot Accuracy, Precision, Recall, F1-Score
metrics = ["precision", "recall", "f1-score"]
plt.figure(figsize=(10, 5))

for metric in metrics:
    color = "blue" if metric == "accuracy" else None  # Ensure accuracy is blue
    plt.plot(results.keys(), [results[m][metric] for m in results], marker="o", label=metric, color=color)


plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.title("Performance Comparison of Models (Including Accuracy)")
plt.legend()
plt.grid(True)
plt.show()

# Bar Graph for Accuracy Comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]

plt.bar(model_names, accuracies, color="blue")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different Models")
plt.ylim(0, 1)  # Ensure y-axis is between 0 and 1
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()