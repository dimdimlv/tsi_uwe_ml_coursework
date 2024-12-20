import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from src.data_preprocessing import apply_smote


def random_forest_experiment(data_path, results_path, experiment_name, rf_params, smote=True):
    """
    Run a Random Forest experiment with specified configurations.

    Args:
        data_path (str): Path to the dataset file.
        results_path (str): Path to save the experiment results.
        experiment_name (str): Name of the experiment for identification.
        rf_params (dict): Parameters for the Random Forest model.
        smote (bool): Whether to apply SMOTE to balance the dataset.

    Returns:
        None
    """
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Default'])
    y = df['Default']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Apply SMOTE to balance the training data if enabled
    if smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # Train Random Forest with specified parameters
    rf_model = RandomForestClassifier(**rf_params, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predictions and probabilities
    y_test_prob = rf_model.predict_proba(X_test)[:, 1]
    y_test_pred = rf_model.predict(X_test)

    # Compute evaluation metrics
    roc_auc = roc_auc_score(y_test, y_test_prob)
    classification_metrics = classification_report(y_test, y_test_pred, output_dict=True)

    # Precision-Recall Curve and threshold optimization
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_idx = f1_scores.argmax()
    best_threshold = thresholds[best_threshold_idx]

    # Save experiment results
    experiment_results = {
        "experiment_name": experiment_name,
        "parameters": rf_params,
        "roc_auc": roc_auc,
        "classification_report": classification_metrics,
        "best_threshold": best_threshold
    }

    with open(results_path, 'a') as file:
        file.write(json.dumps(experiment_results) + '\n')

    print(f"Experiment '{experiment_name}' completed. Results saved to {results_path}")


if __name__ == "__main__":
    # Define the dataset path
    data_path = '../data/modified/cw_final.csv'
    results_path = '../results/random_forest_results.json'

    # Experiment configurations
    experiments = [
        {
            "name": "Baseline Random Forest",
            "params": {"n_estimators": 100, "max_depth": None, "class_weight": "balanced"}
        },
        {
            "name": "RF with Limited Depth",
            "params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "name": "RF with Stricter Branching",
            "params": {"n_estimators": 200, "max_depth": None, "min_samples_split": 5}
        },
        {
            "name": "RF with Custom Weights",
            "params": {"n_estimators": 300, "max_depth": 20, "class_weight": {0: 1, 1: 3}}
        },
        {
            "name": "RF with Feature Subset Selection",
            "params": {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced"}
        }
    ]

    # Run each experiment
    for exp in experiments:
        random_forest_experiment(
            data_path=data_path,
            results_path=results_path,
            experiment_name=exp["name"],
            rf_params=exp["params"],
            smote=True  # Enable SMOTE for all experiments
        )
