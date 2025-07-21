import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv('C://Users//ashun//OneDrive//Desktop//test//Review.csv')  # adjust path if needed

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract review texts
reviews = df['Review'].tolist()

# Generate SBERT embeddings
text_features = model.encode(reviews, convert_to_numpy=True)

# Optional: Save embeddings to disk
np.save('C://Users//ashun//OneDrive//Desktop//test/sbert_text_features.npy', text_features)

print("Done! Feature shape:", text_features.shape)
