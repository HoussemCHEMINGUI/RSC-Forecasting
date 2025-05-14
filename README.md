# Data Analysis Library

This library provides functions for visualizing and evaluating data, including confusion matrices, bar charts for performance metrics, and regression plots. Additionally, it includes functions for calculating key evaluation metrics such as RMSE, MAE, and R².

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import analysis_lib as al

# Example Confusion Matrix Data
eol_conf_matrix = np.array([[211, 23], [28, 138]])
al.plot_eol_confusion_matrix(eol_conf_matrix)

# Example Transport Confusion Matrix Data
transport_conf_matrix = np.array([
    [59, 6, 2, 10, 3],
    [7, 56, 7, 6, 4],
    [5, 9, 60, 1, 5],
    [0, 5, 5, 64, 6],
    [6, 8, 4, 11, 51]
])
labels = ["Boat", "Bus", "Plane", "Train", "Truck"]
al.plot_transport_confusion_matrix(transport_conf_matrix, labels)

# Example Evaluation Metrics
actual = np.random.normal(30, 5, 400)
predicted = actual + np.random.normal(0, 2.87, 400)
metrics = al.calculate_metrics(actual, predicted)
print(metrics)
```

## Requirements

- numpy
- matplotlib
- seaborn
- scikit-learn
- pandas
