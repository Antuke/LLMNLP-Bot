import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ollama  # Assuming ollama is a custom module for embedding

# Function to embed sentences
def embed(sentence):
    response = ollama.embed("nomic-embed-text", "classification: " + sentence)
    return response['embeddings'][0]

df = pd.read_csv('./bigger_dataset.csv',sep=";")  # Replace with your dataset file path

# Embed all sentences
embeddings = np.array([embed(sentence) for sentence in df['sentence']])
print(f"embedding shape= {embeddings.shape}")
# Apply PCA to reduce dimensions to 3
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

# Get unique labels and assign a color to each label
labels = df['label'].unique()
label_to_color = {label: plt.cm.tab10(i) for i, label in enumerate(labels)}  # Use a colormap for distinct colors

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each point with its corresponding label color
for label in labels:
    # Filter embeddings for the current label
    indices = df['label'] == label
    ax.scatter(
        reduced_embeddings[indices, 0],  # PCA 1
        reduced_embeddings[indices, 1],  # PCA 2
        reduced_embeddings[indices, 2],  # PCA 3
        color=label_to_color[label],      # Assign color based on label
        label=label,                     # Label for legend
        s=50                             # Size of the points
    )

# Labeling the axes
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# Add a legend
ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()