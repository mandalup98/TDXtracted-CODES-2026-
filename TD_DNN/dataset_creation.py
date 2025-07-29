import os
import pandas as pd

# Updated base path for filtered files
base_path = ""

# Mapping of benchmark labels to their corresponding filenames
file_map = {
    "AlexNet": "AlexNet_filtered.csv",
    "VGG": "VGG-19_filtered.csv",
    "ResNet": "ResNet-152_filtered.csv",
    "SqueezeNet": "SqueezeNet1-1_filtered.csv",
    "DenseNet": "DenseNet-201_filtered.csv",
    "GoogleNet": "GoogleNet_filtered.csv",
    "ShuffleNet_v2": "ShuffleNet_v2_filtered.csv",
    "MobileNet_v2": "MobileNet_v2_filtered.csv",
    "Inception_v3": "Inception_v3_filtered.csv",
}

def create_samples(values, label, chunk_size=50):
    num_samples = len(values) // chunk_size
    return [
        {"features": values[i*chunk_size:(i+1)*chunk_size], "label": label}
        for i in range(num_samples)
    ]

all_samples = []

for bench, filename in file_map.items():
    file_path = os.path.join(base_path, filename)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            values = df.values.flatten()
            values = [float(v) for v in values if str(v).replace('.', '', 1).isdigit()]
            samples = create_samples(values, bench)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Failed to process {bench}: {e}")
    else:
        print(f"File not found for: {bench}")

# Convert to DataFrame and save
df_dataset = pd.DataFrame(all_samples)
df_dataset.to_csv("DNN_filtered_dataset.csv", index=False)
