# main.py
import time
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.models import get_logistic_regression, get_random_forest, get_svm, get_knn, get_naive_bayes, get_xgboost
from src.train import train_and_predict, evaluate_model
from src.evaluate import print_evaluation_metrics

# Load and preprocess data
print("Loading and preprocessing data...")
df = load_data("data/cw_modified.csv")  # Replace with actual file path
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
print("Data loaded and preprocessed successfully.\n")

# Dictionary of models
models = {
    "Logistic Regression": get_logistic_regression(),
    "Random Forest": get_random_forest(),
    #"SVM": get_svm(),
    "KNN": get_knn(),
    "Naive Bayes": get_naive_bayes(),
    "XGBoost": get_xgboost()
}

# Train and evaluate each model separately
for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()

    # Train and predict
    y_pred, y_pred_proba = train_and_predict(model, X_train, X_test, y_train)

    # Calculate metrics
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    print_evaluation_metrics(model_name, metrics)

    # Display elapsed time for training and evaluation
    elapsed_time = time.time() - start_time
    print(f"{model_name} training and evaluation completed in {elapsed_time:.2f} seconds.\n")