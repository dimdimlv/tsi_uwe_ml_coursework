import json
import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_results(results_path):
    """
    Load experiment results from a JSON file.

    Args:
        results_path (str): Path to the results file.

    Returns:
        pd.DataFrame: Experiment results as a DataFrame.
    """
    with open(results_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


def plot_metrics(results_df, metrics, title="Experiment Metrics Comparison"):
    """
    Plot bar charts for specified metrics.

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.
        metrics (list): List of metric names to plot.
        title (str): Title for the plot.
    """
    experiment_names = results_df['experiment_name']
    num_metrics = len(metrics)

    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    x = range(len(experiment_names))

    for i, metric in enumerate(metrics):
        metric_values = results_df[metric]
        plt.bar(
            [pos + i * bar_width for pos in x],
            metric_values,
            width=bar_width,
            label=metric
        )

    plt.xticks([pos + bar_width * (num_metrics - 1) / 2 for pos in x], experiment_names, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Experiments")
    plt.ylabel("Scores")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load results
    results_path = '../results/random_forest_results.json'
    results_df = load_experiment_results(results_path)

    # Extract relevant metrics
    results_df['F1-Score (Default)'] = results_df['classification_report'].apply(lambda x: x['1']['f1-score'])
    results_df['Accuracy'] = results_df['classification_report'].apply(lambda x: x['accuracy'])

    # Plot metrics
    plot_metrics(
        results_df,
        metrics=['roc_auc', 'F1-Score (Default)', 'Accuracy'],
        title="Random Forest Experiment Metrics"
    )