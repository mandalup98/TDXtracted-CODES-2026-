import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

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

# Process both datasets
results_10, total_pairs_10 = process_dataset(cifar10_path)
results_100, total_pairs_100 = process_dataset(cifar100_path)

# Plot
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(7,5))
bars1 = ax.bar(x - width/2, results_10, width, label='CIFAR-10', facecolor='lightpink', edgecolor='purple', hatch='//')
bars2 = ax.bar(x + width/2, results_100, width, label='CIFAR-100', facecolor='lightblue', edgecolor='blue', hatch='\\\\')

for bars, res in zip([bars1, bars2], [results_10, results_100]):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, f"{res[i]}", ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Number of Distinguishable Class Pairs')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, max(results_10 + results_100)*1.2 if (results_10 + results_100) else 1)
ax.legend()
ax.set_title('Class Pair Distinguishability: CIFAR-10 vs CIFAR-100')

plt.tight_layout()
plt.savefig('cifar10_cifar100_distinguishability_comparison.pdf', dpi=300)
plt.show()
