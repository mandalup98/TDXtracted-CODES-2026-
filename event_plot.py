import matplotlib.pyplot as plt
import numpy as np

# Event categories
events = ['pipeline', 'cache', 'frontend', 'memory','floating point']

# Data
total = [116, 57, 35, 25, 21]
non_zero = [90, 47, 33, 5, 4]
distinguish = [40, 21, 15, 4, 0]

# Y positions and bar width
y = np.arange(len(events))
bar_width = 0.26

# Create the plot
plt.figure(figsize=(4.5, 2.5))

# Draw bars
bars_total = plt.barh(y - bar_width, total, height=bar_width, label='Total Events', color='lightgreen', edgecolor='gray')
bars_non_zero = plt.barh(y, non_zero, height=bar_width, label='Non-Zero Events', color='yellow', edgecolor='gray')
bars_distinguish = plt.barh(y + bar_width, distinguish, height=bar_width, label='Ability to Disntinguish', color= 'red', edgecolor='gray')

# Add counts as labels on bars
for bars in [bars_total, bars_non_zero, bars_distinguish]:
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{int(width)}', va='center', fontsize=8)

# Set labels and title
plt.yticks(y, events)
plt.xlabel("Number of performance counter events")
plt.legend(loc='upper right')
plt.grid(axis='x', linestyle=':', alpha=0.7)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.xlim(0,130)

plt.tight_layout()
plt.savefig("ALL_PMC.pdf", dpi=300, bbox_inches='tight')
plt.show()
