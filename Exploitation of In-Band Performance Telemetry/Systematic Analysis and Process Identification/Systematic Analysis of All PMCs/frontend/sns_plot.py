
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
    "uops_decoded.dec0_uops", "uops_dispatched.port_0", "uops_dispatched.port_1",
    "uops_dispatched.port_2_3_10", "uops_dispatched.port_4_9", "uops_dispatched.port_5_11",
    "uops_dispatched.port_6", "uops_dispatched.port_7_8", "uops_executed.core",
    "uops_executed.core_cycles_ge_1", "uops_executed.core_cycles_ge_2", "uops_executed.core_cycles_ge_3",
    "uops_executed.core_cycles_ge_4", "uops_executed.cycles_ge_1", "uops_executed.cycles_ge_2",
    "uops_executed.cycles_ge_3", "uops_executed.cycles_ge_4", "uops_executed.stall_cycles",
    "uops_executed.stalls", "uops_executed.thread", "uops_executed.x87", "uops_issued.any",
    "uops_retired.cycles", "uops_retired.heavy", "uops_retired.ms", "uops_retired.slots",
    "uops_retired.stall_cycles", "uops_retired.stalls"
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
