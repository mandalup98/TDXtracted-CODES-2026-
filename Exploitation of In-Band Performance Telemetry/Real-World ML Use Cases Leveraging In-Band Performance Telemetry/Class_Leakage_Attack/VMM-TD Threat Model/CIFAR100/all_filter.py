import pandas as pd
import os

base_path = ""
benchmarks = ["class0","class1","class2","class3","class4","class5","class6","class7","class8","class9","class10","class11","class12","class13","class14","class15","class16","class17","class18","class19","class20","class21","class22","class23","class24","class25","class26","class27","class28","class29","class30","class31","class32","class33","class34","class35","class36","class37","class38","class39","class40","class41","class42","class43","class44","class45","class46","class47","class48","class49","class50","class51","class52","class53","class54","class55","class56","class57","class58","class59","class60","class61","class62","class63","class64","class65","class66","class67","class68","class69","class70","class71","class72","class73","class74","class75","class76","class77","class78","class79","class80","class81","class82","class83","class84","class85","class86","class87","class88","class89","class90","class91","class92","class93","class94","class95","class96","class97","class98","class99"]

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
