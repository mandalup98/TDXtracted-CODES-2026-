import os
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
import matplotlib.pyplot as plt

# Base path where each model's directory is located
base_path = ""

# Automatically detect model directories
model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Load event names from the first model
first_model_path = os.path.join(base_path, model_dirs[0])
event_files = [f for f in os.listdir(first_model_path) if f.endswith(".csv")]
event_names = [f.replace(".csv", "") for f in event_files]

# Process each event
for event in event_names:
    event_data = {}
    
    # Load data from all models for this event
    for model in model_dirs:
        file_path = os.path.join(base_path, model, f"{event}.csv")
        if os.path.exists(file_path):
            try:
                values = pd.read_csv(file_path).squeeze().dropna().values
                if len(values) > 0:
                    event_data[model] = values
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if len(event_data) < 2:
        continue  # Need at least 2 models for comparison

    print(f"\n=== T-Test Results for Event: {event} ===")
    pairs = list(combinations(event_data.keys(), 2))
    t_values = []
    p_values = []
    labels = []

    for model1, model2 in pairs:
        data1 = event_data[model1]
        data2 = event_data[model2]

        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
        t_values.append(t_stat)
        p_values.append(p_val)
        label = f"{model1} vs {model2}"
        labels.append(label)
        print(f"{label:25} | t = {t_stat:.3f}, p = {p_val:.4e}")

