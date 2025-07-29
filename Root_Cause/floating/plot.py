import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Benchmarks and Events
benchmarks = [
    "dhry2reg", "whetstone", "execl", "fstime", "pipe",
    "spawn", "shell16"
]

events = [
    "arith.fpdiv_active",
    "assists.fp",
    "assists.sse_avx_mix",
    "fp_arith_dispatched.port_0",
    "fp_arith_dispatched.port_1",
    "fp_arith_dispatched.port_5",
    "fp_arith_inst_retired.128b_packed_double",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.256b_packed_double",
    "fp_arith_inst_retired.256b_packed_single",
    "fp_arith_inst_retired.512b_packed_double",
    "fp_arith_inst_retired.512b_packed_single",
    "fp_arith_inst_retired.scalar_double",
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired2.128b_packed_half",
    "fp_arith_inst_retired2.256b_packed_half",
    "fp_arith_inst_retired2.512b_packed_half",
    "fp_arith_inst_retired2.complex_scalar_half",
    "fp_arith_inst_retired2.scalar",
    "fp_arith_inst_retired2.scalar_half",
    "fp_arith_inst_retired2.vector"
]

base_path = ""

# Hatching patterns (repeats if more events)
hatch_patterns = ['/', '\\', '-', '-|', '+', '|||', 'o', '.', '.', 'P','*']

# Plot in chunks
chunk_size = 7  # Number of events per figure
num_chunks = (len(events) + chunk_size - 1) // chunk_size

for chunk_index in range(num_chunks):
    start = chunk_index * chunk_size
    end = min(start + chunk_size, len(events))
    chunk_events = events[start:end]

    # Prepare data array
    data = np.zeros((len(chunk_events), len(benchmarks)))

    for e_idx, event in enumerate(chunk_events):
        for b_idx, bench in enumerate(benchmarks):
            file_path = os.path.join(base_path, bench, f"{event}.csv")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        values = [int(line.strip()) for line in f if line.strip().isdigit()]
                        if values:
                            data[e_idx, b_idx] = values[-1]  # Use last value
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                data[e_idx, b_idx] = 0

    # Plotting this chunk
    bar_width = 0.8 / len(chunk_events)
    x = np.arange(len(benchmarks))

    # Use seaborn color palette
    palette = sns.color_palette("pastel", len(chunk_events))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, (event, pattern) in enumerate(zip(chunk_events, hatch_patterns)):
        ax.bar(x + i * bar_width, data[i], width=bar_width,
               label=event,
               color=palette[i],     # Seaborn color
               hatch=pattern,
               edgecolor='black')

    # Customize axis
    ax.set_xticks(x + bar_width * (len(chunk_events) - 1) / 2)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize = 12)
    ax.set_ylabel("Pipeline Events")
    ax.set_xlabel("Unixbench")
    ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(top=0.65)  # Shrinks the actual plot height
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),  # Y > 1.0 places legend outside
        ncol=4,
        fontsize = 10
    )
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.7)

    plt.tight_layout()
    output_file = os.path.join(base_path, f"floating_{chunk_index+1}.pdf")
    plt.savefig(output_file)
    plt.show()
