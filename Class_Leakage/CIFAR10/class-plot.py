import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import string
import math

metrics = [
    "cpu_clk_unhalted.thread_p.csv",
    "cycle_activity.cycles_mem_any.csv",
    "idq_uops_not_delivered.core.csv",
    "l1d_pend_miss.pending.csv",
    "memory_activity.stalls_l1d_miss.csv",
    "memory_activity.stalls_l2_miss.csv",
    "uops_retired.stalls.csv",
]

metric_labels = [m.replace('.csv', '') for m in metrics]
letter_labels = list(string.ascii_uppercase[:len(metrics)])

# Paths
cifar10_path = ""
cifar100_path = ""

def process_dataset(base_path):
    class_dirs = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    class_data = {}
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        metric_data = {}
        complete = True
        for metric_file in metrics:
            file_path = os.path.join(class_dir, metric_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty and df.shape[1] >= 1:
                        metric_data[metric_file] = df.iloc[:, -1].values
                    else:
                        complete = False
                except:
                    complete = False
            else:
                complete = False
        if complete:
            class_data[class_name] = metric_data
    # Pairwise t-test
    results = []
    class_names = list(class_data.keys())
    total_pairs = len(class_names) * (len(class_names) -1 ) // 2
    for metric_file in metrics:
        distinguishable_count = 0
        for i in range(len(class_names)):
            for j in range(i+1, len(class_names)):
                c1 = class_data[class_names[i]][metric_file]
                c2 = class_data[class_names[j]][metric_file]
                if len(c1) > 1 and len(c2) > 1:
                    t_stat, _ = ttest_ind(c1, c2, equal_var=False)
                    if abs(t_stat) > 4.5:
                        distinguishable_count += 1
        results.append(distinguishable_count)
    return results, total_pairs

# Process datasets
results_10, total_pairs_10 = process_dataset(cifar10_path)
results_100, total_pairs_100 = process_dataset(cifar100_path)

# Plot
x = np.arange(len(metrics))
width = 0.40

fig, ax1 = plt.subplots(figsize=(7,3))

# Primary y-axis (CIFAR-10)
bars1 = ax1.bar(x - width/2, results_10, width, label='CIFAR-10', facecolor='lightpink', edgecolor='crimson', hatch='//')
ax1.set_ylabel('CIFAR-10 Distinguishable Pairs', color='crimson', fontsize=12)
ax1.tick_params(axis='y', labelcolor='crimson')
ax1.set_ylim(0, max(results_10)*1.3 if results_10 else 1)

# Secondary y-axis (CIFAR-100)
ax2 = ax1.twinx()
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

bars2 = ax2.bar(x + width/2, results_100, width, label='CIFAR-100', facecolor='lightblue', edgecolor='blue', hatch='\\\\')
ax2.set_ylabel('CIFAR-100 Distinguishable Pairs', color='blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, max(results_100)*1.3 if results_100 else 1)

# Annotations
for bars, res, ax in zip([bars1, bars2], [results_10, results_100], [ax1, ax2]):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{res[i]}", ha='center', va='bottom', fontsize=8)

# X-axis and grid
ax1.set_xticks(x)
ax1.set_xticklabels(letter_labels, rotation=0, ha='center', fontsize=13)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Legend
fig.legend(['CIFAR-10', 'CIFAR-100'], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.99), fontsize=14, frameon=True, edgecolor='gray')

# Add mapping text in 3 columns below plot



plt.tight_layout()
plt.savefig('cifar10_distinguishability_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()
