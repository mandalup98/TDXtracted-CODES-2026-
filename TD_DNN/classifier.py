
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the updated dataset
df = pd.read_csv("DNN_filtered_dataset.csv")

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train the model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
accuracy = mlp.score(X_test, y_test)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_normalized,
    annot=True,
    fmt='.2f',
    cmap='PuRd',
    xticklabels=label_names,
    yticklabels=label_names,
    cbar_kws={'format': '%.0f%%'},
    annot_kws={"size": 12}
)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=90, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.savefig("TD_DNN_filtered_confusion_matrix.pdf", dpi=500, bbox_inches='tight')
plt.show()
