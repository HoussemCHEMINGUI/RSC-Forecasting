import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Confusion Matrix - EOL Status
def plot_eol_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix - EOL Status")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(ticks=[0.5, 1.5], labels=["False", "True"])
    plt.yticks(ticks=[0.5, 1.5], labels=["False", "True"], rotation=0)
    plt.tight_layout()
    plt.show()

# Confusion Matrix - Transport Mode
def plot_transport_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Oranges", fmt="d", cbar=False)
    plt.title("Confusion Matrix - Transport Mode")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)
    plt.tight_layout()
    plt.show()

# Bar Chart - Performance Metrics
def plot_metrics_bar_chart(metrics_data):
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Transport Mode").plot(kind="bar", figsize=(10, 6))
    plt.title("Performance Metrics - Transport Mode")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Regression Plot
def plot_regression(actual, predicted, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], linestyle="--", color="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

# Evaluation Metrics
def calculate_metrics(actual, predicted):
    rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }
