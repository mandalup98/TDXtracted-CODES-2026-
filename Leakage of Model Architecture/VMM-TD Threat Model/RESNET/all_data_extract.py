import os
import pandas as pd
import re

# Set input/output directories and benchmark list
base_path = ""
benchmarks = [ "ResNet-18", "ResNet-34","ResNet-50","ResNet-101","ResNet-152"]

# Regex to parse the text lines
pattern = re.compile(r"^\s*(\d+\.\d+)\s+(<not counted>|[\d,]+)\s+(\S+)\s*(?:\(\d+\.?\d*%\))?$")

for benchmark in benchmarks:
    input_file = os.path.join(base_path, f"{benchmark}.txt")
    output_file = os.path.join(base_path, f"{benchmark}.csv")

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        continue

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    records = []
    current_record = {}

    for line in lines:
        match = pattern.match(line)
        if match:
            time, value, event = match.groups()
            if current_record.get("time") != time:
                if current_record:
                    records.append(current_record)
                current_record = {"time": time}
            current_record[event] = 0 if value == "<not counted>" else int(value.replace(",", ""))

    # Add the last record
    if current_record:
        records.append(current_record)

    df = pd.DataFrame(records)

    print(f"{benchmark}: DataFrame has {len(df)} rows and columns: {list(df.columns)}")

    if "time" in df.columns:
        df.drop(columns=["time"], inplace=True)

    df.to_csv(output_file, index=False)
    print(f"{benchmark}: Processed and saved to {output_file}")
