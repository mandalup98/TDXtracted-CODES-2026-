import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("DENSENET_br_misp_retired.all_branches.csv")

# Convert stringified list to numerical array
df["features"] = df["features"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
X = np.stack(df["features"].values)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
label_names = label_encoder.classes_

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predictions and confusion matrix
y_pred = mlp.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_normalized, 
    annot=True, 
    fmt='.2f', 
    cmap='BuPu', 
    xticklabels=label_names, 
    yticklabels=label_names, 
    cbar_kws={'format': '%.0f%%'},
    annot_kws={"size": 20}
)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()

# Save to PDF
plt.savefig("DENSENET_br_misp_retired.all_branches.pdf", dpi=500, bbox_inches='tight')
plt.show()
