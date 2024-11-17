import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('../data/modified/cw_final.csv')
X = df.drop(columns=['Default'])
y = df['Default']

# Split data into train, validation, and test sets (60%-20%-20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply SMOTE to the training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust sampling_strategy as needed
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train logistic regression with custom class weights and regularization
model = LogisticRegression(class_weight={0: 1, 1: 3}, C=0.1, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities on the validation set
y_val_prob = model.predict_proba(X_val)[:, 1]

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Validation Set)")
plt.legend()
plt.grid()
plt.show()

# Find the best threshold (maximize F1-Score)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_idx = f1_scores.argmax()
best_threshold = thresholds[best_threshold_idx]

print(f"Best Threshold: {best_threshold}")
print(f"Precision at Best Threshold: {precision[best_threshold_idx]}")
print(f"Recall at Best Threshold: {recall[best_threshold_idx]}")

# Evaluate the model on the test set using the best threshold
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

# Compute metrics on the test set
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_prob)
report = classification_report(y_test, y_test_pred)

print("\nTest Set Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score (Default): {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", report)