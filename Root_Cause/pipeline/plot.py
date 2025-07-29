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
    # "cpu_clk_unhalted.c01",
    # "cpu_clk_unhalted.c02",
    # "cpu_clk_unhalted.c0_wait",
    "cpu_clk_unhalted.distributed",
    # "cpu_clk_unhalted.one_thread_active",
    # "cpu_clk_unhalted.pause",
    # "cpu_clk_unhalted.pause_inst",
    # "cpu_clk_unhalted.ref_distributed",
    "cpu_clk_unhalted.ref_tsc",
    "cpu_clk_unhalted.thread",
    "cpu_clk_unhalted.thread_p",
    # "cycle_activity.cycles_l1d_miss",
    # "cycle_activity.cycles_l2_miss",
    "cycle_activity.cycles_mem_any",
    # "cycle_activity.stalls_l1d_miss",
    # "cycle_activity.stalls_l2_miss",
    "cycle_activity.stalls_total",
    # "exe_activity.1_ports_util",
    # "exe_activity.2_ports_util",
    # "exe_activity.3_ports_util",
    # "exe_activity.4_ports_util",
    # "exe_activity.bound_on_loads",
    # "exe_activity.bound_on_stores",
    # "exe_activity.exe_bound_0_ports",
    # "inst_decoded.decoders",
    "inst_retired.any",
    # "inst_retired.any_p",
    # "inst_retired.macro_fused",
    # "inst_retired.nop",
    # "inst_retired.prec_dist",
    # "inst_retired.rep_iteration",
    # "int_misc.clear_resteer_cycles",
    # "int_misc.mba_stalls",
    # "int_misc.recovery_cycles",
    # "int_misc.unknown_branch_cycles",
    # "int_misc.uop_dropping",
    # "int_vec_retired.128bit",
    # "int_vec_retired.256bit",
    # "int_vec_retired.add_128",
    # "int_vec_retired.add_256",
    # "int_vec_retired.mul_256",
    # "int_vec_retired.shuffles",
    # "int_vec_retired.vnni_128",
    # "int_vec_retired.vnni_256",
    # "ld_blocks.address_alias",
    # "ld_blocks.no_sr",
    # "ld_blocks.store_forward",
    # "load_hit_prefetch.swpf",
    # "lsd.cycles_active",
    # "lsd.cycles_ok",
    # "lsd.uops",
    # "machine_clears.count",
    # "machine_clears.smc",
    # "misc2_retired.lfence",
    # "misc_retired.lbr_inserts",
    # "resource_stalls.sb",
    # "resource_stalls.scoreboard",
    # "topdown.backend_bound_slots",
    # "topdown.bad_spec_slots",
    # "topdown.br_mispredict_slots",
    # "topdown.memory_bound_slots",
    # "topdown.slots",
    # "topdown.slots_p",
    # "uops_decoded.dec0_uops",
    # "uops_dispatched.port_0",
    # "uops_dispatched.port_1",
    # "uops_dispatched.port_2_3_10",
    # "uops_dispatched.port_4_9",
    # "uops_dispatched.port_5_11",
    # "uops_dispatched.port_6",
    # "uops_dispatched.port_7_8",
    # "uops_executed.core",
    # "uops_executed.core_cycles_ge_1",
    # "uops_executed.core_cycles_ge_2",
    # "uops_executed.core_cycles_ge_3",
    # "uops_executed.core_cycles_ge_4",
    # "uops_executed.cycles_ge_1",
    # "uops_executed.cycles_ge_2",
    # "uops_executed.cycles_ge_3",
    # "uops_executed.cycles_ge_4",
    "uops_executed.stall_cycles",
    "uops_executed.stalls",
    "uops_executed.thread",
    # "uops_executed.x87",
    "uops_issued.any",
    # "uops_retired.cycles",
    # "uops_retired.heavy",
    # "uops_retired.ms",
    "uops_retired.slots",
    "uops_retired.stall_cycles",
    "uops_retired.stalls"
]

base_path = ""

# Hatching patterns (repeats if more events)
hatch_patterns = ['/', '\\', '-', '-|', '+', '|||', 'o', '.', '.', 'P','*']

# Plot in chunks
chunk_size = 8 # Number of events per figure
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
    palette = sns.color_palette("Set2", len(chunk_events))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i, (event, pattern) in enumerate(zip(chunk_events, hatch_patterns)):
        ax.bar(x + i * bar_width, data[i], width=bar_width,
               label=event,
               color=palette[i],     # Seaborn color
               hatch=pattern,
               edgecolor='black')

    # Customize axis
    ax.set_xticks(x + bar_width * (len(chunk_events) - 1) / 2)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize = 12)
    ax.set_ylabel("pipeline events")
    ax.set_xlabel("Unixbench")
    ax.tick_params(axis='y', labelsize=12)
    fig.subplots_adjust(top=0.65)  # Shrinks the actual plot height
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),  # Y > 1.0 places legend outside
        ncol=3,
        fontsize = 9.2
    )
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.7)

    plt.tight_layout()
    output_file = os.path.join(base_path, f"pipeline_{chunk_index+1}.pdf")
    plt.savefig(output_file)
    plt.show()
