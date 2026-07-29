import os
import pandas as pd
import re

# Set input/output directories and benchmark list
base_path = ""
benchmarks = ["class0","class1","class2","class3","class4","class5","class6","class7","class8","class9","class10","class11","class12","class13","class14","class15","class16","class17","class18","class19","class20","class21","class22","class23","class24","class25","class26","class27","class28","class29","class30","class31","class32","class33","class34","class35","class36","class37","class38","class39","class40","class41","class42","class43","class44","class45","class46","class47","class48","class49","class50","class51","class52","class53","class54","class55","class56","class57","class58","class59","class60","class61","class62","class63","class64","class65","class66","class67","class68","class69","class70","class71","class72","class73","class74","class75","class76","class77","class78","class79","class80","class81","class82","class83","class84","class85","class86","class87","class88","class89","class90","class91","class92","class93","class94","class95","class96","class97","class98","class99"]

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
