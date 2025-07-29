
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global seaborn style
sns.set_style("whitegrid")
sns.set_context("talk")

# Base directory
base_path = ""

# Benchmarks
benchmarks = [
    "dhry2reg", "whetstone", "execl", "fstime", "pipe",
    "spawn", "syscall", "context1", "shell16"
]

# Events to plot
events = [
    "mem_inst_retired.all_loads", "mem_inst_retired.all_stores", "mem_inst_retired.any",
    "mem_inst_retired.lock_loads", "mem_inst_retired.split_loads", "mem_inst_retired.split_stores",
    "mem_inst_retired.stlb_miss_loads", "mem_inst_retired.stlb_miss_stores", "mem_load_completed.l1_miss_any",
    "mem_load_l3_hit_retired.xsnp_fwd", "mem_load_l3_hit_retired.xsnp_miss", "mem_load_l3_hit_retired.xsnp_no_fwd",
    "mem_load_l3_hit_retired.xsnp_none", "mem_load_l3_miss_retired.local_dram", "mem_load_l3_miss_retired.remote_dram",
    "mem_load_l3_miss_retired.remote_fwd", "mem_load_l3_miss_retired.remote_hitm", "mem_load_l3_miss_retired.remote_pmm",
    "mem_load_misc_retired.uc", "mem_load_retired.fb_hit", "mem_load_retired.l1_hit", "mem_load_retired.l1_miss",
    "mem_load_retired.l2_hit", "mem_load_retired.l2_miss", "mem_load_retired.l3_hit", "mem_load_retired.l3_miss",
    "mem_load_retired.local_pmm", "mem_store_retired.l2_hit", "mem_uop_retired.any", "longest_lat_cache.miss"
]

# Color palette
palette = sns.color_palette("husl", len(benchmarks))  # Unique color for each benchmark

# Generate KDE plots for each event
for event in events:
    plt.figure(figsize=(5, 4))

    for i, bench in enumerate(benchmarks):
        file_path = os.path.join(base_path, bench, f"{event}.csv")

        if os.path.exists(file_path):
            try:
                # Read data from CSV, clean and filter numeric lines
                with open(file_path, 'r') as f:
                    values = [int(line.strip()) for line in f if line.strip().isdigit()]

                if values:
                    sns.kdeplot(
                        values,
                        label=bench,
                        fill=True,
                        alpha=0.3,
                        bw_adjust=0.8,
                        color=palette[i]
                    )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Plot enhancements
    plt.xlabel(event, fontsize=8)
    plt.ylabel("Density", fontsize=8)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.7)
    # plt.legend(title="Benchmark", bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=8)
    plt.legend(
    title="Benchmark",
    fontsize=7,
    title_fontsize=8,
    loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    x_min, x_max = plt.xlim()

    # Compute two evenly spaced internal ticks
    tick1 = x_min + (x_max - x_min) / 3
    tick2 = x_min + 2 * (x_max - x_min) / 3

    # Set x-axis ticks to just these two
    plt.xticks([int(tick1), int(tick2)], fontsize=8)
    plt.yticks(fontsize = 8)

    # Save the plot
    output_file = os.path.join(base_path, f"{event.replace('.', '_')}_kde.pdf")
    plt.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close()
