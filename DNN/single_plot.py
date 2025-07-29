import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style
sns.set(style="whitegrid", context="notebook", font_scale=1.1)

# Base directory and benchmarks
base_path = ""
benchmarks = ["alexnet", "vgg", "resnet", "squeezenet", "densenet", "googlenet", "shufflenet", "mobilenet", "inception"]

# Common events
common_events = [
    "br_misp_retired.all_branches",
    "cpu_clk_unhalted.ref_tsc",
    #"cpu_clk_unhalted.thread_p",
    #"cycle_activity.cycles_mem_any",
    #"idq_uops_not_delivered.core",
    #"l1d_pend_miss.pending",
    # "mem_inst_retired.any",
    # "mem_uop_retired.any",
    "memory_activity.stalls_l1d_miss",
    "memory_activity.stalls_l2_miss"
    # "uops_issued.any",
    #"uops_retired.stalls",
]

# Output file
output_file = os.path.join(base_path, "all_events_grid.pdf")

# Setup grid: 4 plots per row
ncols = 4
nrows = (len(common_events) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

# Markers and colors
marker_styles = ['o', 's', 'v', '^', '<', '>', 'P', 'X', 'D', '*', 'p', 'h', '.', '1', '8']
palette = sns.color_palette("tab20", n_colors=len(benchmarks))

# Track plotted handles/labels for shared legend
handles, labels = None, None

# Loop through events and subplot axes
for idx, event in enumerate(common_events):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]

    for bidx, benchmark in enumerate(benchmarks):
        event_file = os.path.join(base_path, benchmark, f"{event}.csv")
        if not os.path.exists(event_file):
            print(f"⚠️ Skipping {benchmark}: {event}.csv not found")
            continue

        try:
            df = pd.read_csv(event_file)
            values = df[event] if event in df.columns else df.iloc[:, 0]
            x_vals = list(range(len(values)))

            line, = ax.plot(
                x_vals,
                values,
                label=benchmark,
                color=palette[bidx % len(palette)],
                marker=marker_styles[bidx % len(marker_styles)],
                markersize=3,
                linewidth=0.8,
                alpha=0.8
            )
        except Exception as e:
            print(f"❌ Error processing {event_file}: {e}")
            continue

    ax.set_xlabel("Samples", fontsize=9)
    ax.set_ylabel(event, fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, 150)

    # Store legend handles/labels from one subplot
    if handles is None and 'line' in locals():
        handles, labels = ax.get_legend_handles_labels()

# Remove any empty axes
for idx in range(len(common_events), nrows * ncols):
    fig.delaxes(axes[idx // ncols][idx % ncols])

# Add shared legend above all subplots
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.00),
    ncol=13,
    fontsize=12,
    frameon=True,
    edgecolor='gray',
    fancybox=True
)

plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for legend at top
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved combined grid plot to: {output_file}")
