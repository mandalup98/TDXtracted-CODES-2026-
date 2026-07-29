import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style
sns.set(style="whitegrid", context="notebook", font_scale=1.1)

# Base directory and benchmarks
benchmarks = ["AttackerTD_process","AttackerTD+VictimTD_process"]
#benchmarks = ["AttackerVMM_process","AttackerVMM+VictimTD_process"]

# Common events
common_events = [
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "slots"
]

# Plot output folder
plot_dir = os.path.join(base_path, "TD_plot")
os.makedirs(plot_dir, exist_ok=True)

# Markers and colors
marker_styles = ['o', 's', 'v', '^', '<', '>', 'P', 'X', 'D', '*', 'p', 'h', '.', '1', '8']
palette = sns.color_palette("Set1", n_colors=len(benchmarks))

# Plot each event
for event in common_events:
    plt.figure(figsize=(5, 3.5))

    for idx, benchmark in enumerate(benchmarks):
        event_file = os.path.join(base_path, benchmark, f"{event}.csv")
        if not os.path.exists(event_file):
            print(f"⚠️ Skipping {benchmark}: {event}.csv not found")
            continue

        try:
            df = pd.read_csv(event_file)
            values = df[event] if event in df.columns else df.iloc[:, 0]
            x_vals = list(range(len(values)))
            
            # Plot line with markers
            plt.plot(
                x_vals,
                values,
                label=benchmark,
                color=palette[idx % len(palette)],
                marker=marker_styles[idx % len(marker_styles)],
                markersize=4,
                linewidth=1,
                alpha=0.8
            )
        except Exception as e:
            print(f"❌ Error processing {event_file}: {e}")
            continue

    # Style the plot
    plt.xlabel("Samples",fontsize=14)
    plt.ylabel(event,fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=1,
        fontsize=14,
        frameon=True,
        edgecolor='gray',
        fancybox=True
    )
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.xlim(0,150)
    # Save and show
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{event}.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot with lines: {plot_path}")
