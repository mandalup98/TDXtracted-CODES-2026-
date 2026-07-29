import os
import pandas as pd

# Base path and list of benchmarks
base_path = ""
benchmarks = [ "SqueezeNet-1.0","SqueezeNet-1.1"]



def create_samples(values, label, chunk_size=10):
    num_samples = len(values) // chunk_size
    return [
        {"features": values[i*chunk_size:(i+1)*chunk_size], "label": label}
        for i in range(num_samples)
    ]

# Aggregate all benchmark data
all_samples = []

for bench in benchmarks:
    file_path = os.path.join(base_path, bench, "memory_activity.stalls_l1d_miss.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            values = df[1:].astype(int).values.flatten()  # Skip header
            samples = create_samples(values, bench)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Failed to process {bench}: {e}")
    else:
        print(f"File not found for benchmark: {bench}")

# Convert to DataFrame and save
df_dataset = pd.DataFrame(all_samples)
df_dataset.to_csv("SQUEEZENET_memory_activity.stalls_l1d_miss.csv", index=False)  # Optional: CSV for readability
