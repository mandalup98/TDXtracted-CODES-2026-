import pandas as pd
import os

# Define the base path
base_path = ""

# List of benchmark filenames (without .csv extension)
benchmarks = [
    "dhry2reg", "whetstone", "excel", "fstime",
    "pipe", "spawn", "syscall", "context1", "shell16"
]

# Loop through each benchmark
for benchmark in benchmarks:
    input_file = os.path.join(base_path, f"{benchmark}.csv")
    output_file = os.path.join(base_path, f"filtered_{benchmark}.csv")
    
    try:
        # Load CSV
        df = pd.read_csv(input_file)
        
        # Convert to numeric and fill NaNs with 0
        df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Remove all-zero columns
        df_cleaned = df.loc[:, (df_numeric != 0).any(axis=0)]
        
        # Save result
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"✅ Processed: {benchmark} → filtered_{benchmark}.csv")
        
    except Exception as e:
        print(f"❌ Failed: {benchmark} — {str(e)}")
