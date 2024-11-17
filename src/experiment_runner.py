import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
import json

# Load the preprocessed dataset
df = pd.read_csv('../data/modified/cw_final.csv')
X = df.drop(columns=['Default'])
y = df['Default']

# Define updated configurations for experiments
configs = [
    {"id": 1, "resampling": None, "class_weight": "balanced", "C": 1.0, "threshold": 0.5},
    {"id": 2, "resampling": "SMOTE", "class_weight": "balanced", "C": 1.0, "threshold": 0.5},
    {"id": 3, "resampling": None, "class_weight": {0: 1, 1: 3}, "C": 0.1, "threshold": 0.5},
    {"id": 4, "resampling": "SMOTE", "class_weight": {0: 1, 1: 3}, "C": 0.1, "threshold": "optimized"},
    {"id": 5, "resampling": "SMOTE", "class_weight": "balanced", "C": 0.01, "threshold": 0.5},
    {"id": 6, "resampling": "SMOTE", "class_weight": {0: 1, 1: 5}, "C": 0.01, "threshold": "optimized"},
]

# Initialize results
results = []

# Run experiments
for config in configs:
    print(f"Running Experiment {config['id']}...")

    # Apply resampling if specified
    if config["resampling"] == "SMOTE":
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    # Initialize logistic regression
    model = LogisticRegression(class_weight=config["class_weight"], C=config["C"], random_state=42)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Optimize threshold if specified
    if config["threshold"] == "optimized":
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[f1_scores.argmax()]
        print(f"Optimized Threshold for Experiment {config['id']}: {best_threshold}")
        config["threshold"] = best_threshold
        y_pred = (y_pred_prob >= best_threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validation for ROC-AUC
    cv_roc_auc = cross_val_score(model, X_resampled, y_resampled, scoring="roc_auc", cv=5).mean()

    # Store results
    results.append({
        "Experiment ID": config["id"],
        "Resampling": config["resampling"],
        "Class Weight": config["class_weight"],
        "C": config["C"],
        "Threshold": config["threshold"],
        "Accuracy": accuracy,
        "F1-Score (Default)": f1,
        "ROC-AUC": roc_auc,
        "CV ROC-AUC": cv_roc_auc,
        "Classification Report": report,
    })

# Save results to a JSON file
with open("../results/logistic_experiment_results_updated.json", "w") as f:
    json.dump(results, f, indent=4)

print("All experiments completed. Results saved to logistic_experiment_results_updated.json.")