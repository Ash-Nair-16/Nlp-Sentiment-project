import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# === Load Data ===
text_features = np.load('C://Users//ashun//OneDrive//Desktop//test//sbert_text_features.npy')
image_features = np.load('C://Users//ashun//OneDrive//Desktop//test//resnet_image_features.npy')
df = pd.read_csv('C://Users//ashun//OneDrive//Desktop//test//Review.csv')

# === Check Feature Shapes ===
print(f"Text features shape: {text_features.shape}")
print(f"Image features shape: {image_features.shape}")

assert text_features.shape[0] == image_features.shape[0] == len(df), " Mismatch in data length!"

# === Check for Missing Labels ===
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment'] = df['Sentiment'].str.lower().str.strip().map(label_map)

missing_labels = df['Sentiment'].isnull().sum()
print(f"\n Missing or unrecognized labels: {missing_labels}")
assert missing_labels == 0, " Found missing/unmapped sentiment labels!"

# === Check Class Balance ===
class_counts = Counter(df['Sentiment'])
class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

print("\n Class Distribution:")
for k, v in class_counts.items():
    print(f"{class_names[k]}: {v} samples")

# === Plot Class Distribution ===
plt.bar(class_names.values(), [class_counts[0], class_counts[1], class_counts[2]], color=['red', 'gray', 'green'])
plt.title("Class Distribution")
plt.xlabel("Sentiment Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

print("\n Dataset integrity check passed!")
