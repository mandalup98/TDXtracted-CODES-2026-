import os
import numpy as np
import pandas as pd

# Input directory: where the .txt files are
input_files = [
    "Class0.txt",
    "Class1.txt",
    "Class2.txt",
    "Class3.txt",
    "Class4.txt",
    "Class5.txt",
    "Class6.txt",
    "Class7.txt",
    "Class8.txt",
    "Class9.txt"
]
input_base = ""
output_dir = os.path.join(input_base, "filtered")
os.makedirs(output_dir, exist_ok=True)

for fname in input_files:
    fpath = os.path.join(input_base, fname)
    # Read all numbers from the txt file (assuming one value per line)
    with open(fpath, 'r') as f:
        lines = [line.strip() for line in f if line.strip().isdigit()]
        values = np.array([int(x) for x in lines])

    # Calculate median (μ)
    mu = np.median(values)

    # Calculate bounds
    lower = 0.95 * mu
    upper = 1.05 * mu

    # Filter values
    filtered = values[(values >= lower) & (values <= upper)]

    # Save filtered values as CSV
    outname = fname.replace(".txt", "_filtered.csv")
    outpath = os.path.join(output_dir, outname)
    pd.DataFrame(filtered, columns=['Value']).to_csv(outpath, index=False)

    print(f"Filtered {fname}: median={mu}, kept {len(filtered)}/{len(values)} values, saved to {outpath}")
