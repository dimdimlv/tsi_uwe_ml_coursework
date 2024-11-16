# src/evaluate.py
def print_evaluation_metrics(model_name, metrics):
    print(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value != "N/A" else f"{metric}: N/A")
