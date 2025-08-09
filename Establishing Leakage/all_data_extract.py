import os
import pandas as pd
import re

base_path = ""
benchmarks = ["AttackerTD+VictimTD_process", "AttackerTD_process"]

# Matches: time, value, event (before '#')
pattern = re.compile(r"^\s*(\d+\.\d+)\s+([<\d,\.]+)\s+(\S.*?)(?=\s+#|$)")

for benchmark in benchmarks:
    input_file = os.path.join(base_path, f"{benchmark}.txt")
    output_file = os.path.join(base_path, f"{benchmark}.csv")

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        continue

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    records = []
    current_time = None
    current_record = {}

    for line in lines:
        match = pattern.match(line)
        if match:
            time, value, event = match.groups()

            # If time changes, flush previous record
            if current_time != time:
                if current_record:
                    records.append(current_record)
                current_time = time
                current_record = {"time": time}

            numeric_value = 0 if "<not counted>" in value else float(value.replace(",", ""))
            current_record[event.strip()] = numeric_value

    if current_record:
        records.append(current_record)

    df = pd.DataFrame(records)

    print(f"{benchmark}: DataFrame has {len(df)} rows and columns: {list(df.columns)}")

    if "time" in df.columns:
        df.drop(columns=["time"], inplace=True)

    df.to_csv(output_file, index=False)
    print(f"{benchmark}: Processed and saved to {output_file}")
