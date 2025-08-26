import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, RocCurveDisplay
)
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("breast-cancer.csv")  

# Data cleaning
if 'Id' in data.columns:
    data = data.drop(['Id'], axis=1)

# Encode diagnosis (M = 1 malignant, B = 0 benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Check for missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Hyperparameter Tuning with GridSearchCV ----------------
# Logistic Regression
log_reg_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
log_reg_grid = GridSearchCV(LogisticRegression(max_iter=1000), log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train_scaled, y_train)
best_log_reg = log_reg_grid.best_estimator_

# SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_

# ---------------- Decision Tree Hyperparameter Tuning ----------------
dt_params = {
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train_scaled, y_train)
best_dt = dt_grid.best_estimator_

# ---------------- Random Forest Hyperparameter Tuning ----------------
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

# ---------------- Cross-Validation ----------------
log_reg_cv_scores = cross_val_score(best_log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm_cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
dt_cv_scores = cross_val_score(best_dt, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(" Logistic Regression Cross-Validation Accuracy")
print(log_reg_cv_scores)
print("Mean CV Accuracy:", log_reg_cv_scores.mean())

print("\n SVM Cross-Validation Accuracy")
print(svm_cv_scores)
print("Mean CV Accuracy:", svm_cv_scores.mean())

print("\n Decision Tree Cross-Validation Accuracy")
print(dt_cv_scores)
print("Mean CV Accuracy:", dt_cv_scores.mean())

print("\n Random Forest Cross-Validation Accuracy")
print(rf_cv_scores)
print("Mean CV Accuracy:", rf_cv_scores.mean())

# ---------------- Evaluation on Test Set ----------------
y_pred_log = best_log_reg.predict(X_test_scaled)
y_pred_svm = best_svm.predict(X_test_scaled)
y_pred_dt = best_dt.predict(X_test_scaled)
y_pred_rf = best_rf.predict(X_test_scaled)

print("\n Logistic Regression Results (Best Params) ")
print("Best Params:", log_reg_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

print("\n SVM Results (Best Params) ")
print("Best Params:", svm_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

print("\n Decision Tree Results (Best Params) ")
print("Best Params:", dt_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

print("\n=== Random Forest Results (Best Params) ===")
print("Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ---------------- ROC Curve and AUC ----------------
y_score_log = best_log_reg.predict_proba(X_test_scaled)[:, 1]
y_score_svm = best_svm.predict_proba(X_test_scaled)[:, 1]
y_score_dt = best_dt.predict_proba(X_test_scaled)[:, 1]
y_score_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_score_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_score_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, color='purple', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



