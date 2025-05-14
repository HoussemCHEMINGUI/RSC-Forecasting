
import numpy as np
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk, scrolledtext, Toplevel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
)

# 1. Generate realistic, balanced synthetic data
N = 2000
np.random.seed(42)

# --- 1. EOL Status ---
months_since_sale = np.random.randint(6, 61, size=N)
num_returns = np.random.randint(0, 4, size=N)
product_categories = [
    "Food", "Electronics", "Pharma", "Beauty", "Automotive",
    "Textile", "Construction", "Other"
]
categories = np.random.choice(product_categories, size=N)
category_indicators = np.zeros((len(categories), len(product_categories)), dtype=int)
for i, cat in enumerate(categories):
    cat_index = product_categories.index(cat)
    category_indicators[i, cat_index] = 1
# Noisy but correlated target
eol_prob = 1 / (1 + np.exp(-0.18*(months_since_sale-36) + 0.7*(num_returns-1)))
eol_status = (np.random.rand(N) < eol_prob).astype(int)
X_eol = np.column_stack((months_since_sale, num_returns, category_indicators))

# --- 2. Disassembly Duration & Transport Mode ---
materials = [
    'Metallic', 'Cotton', 'Steel', 'Aluminum', 'Wood', 'Plastic',
    'Glass', 'Leather', 'Concrete', 'Stone', 'Copper', 'Bronze', 'Titanium'
]
weight = np.random.uniform(10, 300, N)
hazardous = np.random.binomial(1, 0.25, N)
product_age = np.random.randint(1, 10, N)
material_features = np.zeros((N, len(materials)), dtype=int)
for i in range(N):
    mat_indices = np.random.choice(range(len(materials)), size=random.randint(1, 3), replace=False)
    material_features[i, mat_indices] = 1

# Disassembly duration: depends on weight, hazardous, age, some materials
disassembly_duration = (
    8 + 0.045*weight + 6*hazardous + 0.7*product_age +
    2.5*material_features[:, materials.index('Steel')] +
    2.5*material_features[:, materials.index('Plastic')] +
    np.random.normal(0, 2, N)
).clip(1, None)

# Create DataFrame
data = {
    'Weight': weight,
    'Hazardous': hazardous,
    'Age': product_age,
    'DisassemblyDuration': disassembly_duration
}
for i, mat in enumerate(materials):
    data[f'Material_{mat}'] = material_features[:, i]

df = pd.DataFrame(data)
df.to_csv("synthetic_data.csv", index=False)

print("Synthetic data generated and saved as synthetic_data.csv")
    