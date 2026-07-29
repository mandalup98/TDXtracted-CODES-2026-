import pandas as pd
import os

# Base directory and benchmark list
base_path = ""
benchmarks = ["alexnet", "vgg16", "resnet101", "squeezenet1_1", "densenet169", "googlenet", "shufflenet", "mobilenet", "inception"]


# Process each benchmark file
for benchmark in benchmarks:
    input_file = os.path.join(base_path, f"{benchmark}.csv")
    output_dir = os.path.join(base_path, benchmark)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"❌ Missing file: {input_file}")
        continue

    # Load and optionally trim rows (like iloc[1000:-100] in your example)
    df = pd.read_csv(input_file)
    # if len(df) > 1100:
    #     df = df.iloc[1200:-100]

    # Process each event column
    for column in df.columns:
        col_data = df[column]
        median = col_data.median()
        lower = 0.95 * median
        upper = 1.05 * median

        #filtered = col_data
        filtered = col_data[(col_data >= lower) & (col_data <= upper)]

        output_file = os.path.join(output_dir, f"{column}.csv")
        filtered.to_csv(output_file, index=False, header=[column])

    print(f"✅ Processed {benchmark}: filtered event files saved in {output_dir}")
