import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style
sns.set(style="whitegrid", context="notebook", font_scale=1.1)

# Base directory and benchmarks
base_path = ""
benchmarks = [ "DenseNet-121", "DenseNet-161", "DenseNet-169","DenseNet-201"]
# Common events
common_events = [
    "br_inst_retired.all_branches",
    "cpu_clk_unhalted.ref_tsc",
    "cpu_clk_unhalted.thread_p",
    "cycle_activity.cycles_mem_any",
    "idq_uops_not_delivered.core",
    "l1d_pend_miss.pending",
    "mem_inst_retired.any",
    "mem_uop_retired.any",
    "memory_activity.stalls_l1d_miss",
    "memory_activity.stalls_l2_miss",
    "uops_issued.any",
    "uops_retired.stalls",
    "br_misp_retired.all_branches"
]

# Plot output folder
plot_dir = os.path.join(base_path, "scatter_plots_with_lines")
os.makedirs(plot_dir, exist_ok=True)

# Markers and colors
marker_styles = ['o', 's', 'v', '^', '<', '>', 'P', 'X', 'D', '*', 'p', 'h', '.', '1', '8']
palette = sns.color_palette("tab10", n_colors=len(benchmarks))

# Plot each event
for event in common_events:
    plt.figure(figsize=(8, 3.5))

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
                linewidth=2.5,
                alpha=0.8
            )
        except Exception as e:
            print(f"❌ Error processing {event_file}: {e}")
            continue

    # Style the plot
    plt.xlabel("Samples", fontsize=16)
    plt.ylabel(event, fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=13,
        frameon=True,
        edgecolor='gray',
        fancybox=True
    )
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlim(0,150)
    # Save and show
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{event}.pdf")
    plt.tight_layout(rect=[0, 0, 1, 1.20]) 
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot with lines: {plot_path}")
