import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Features and Model ===
text_feat = np.load("C:/Users/ashun/OneDrive/Desktop/test/sbert_text_features.npy")
image_feat = np.load("C:/Users/ashun/OneDrive/Desktop/test/resnet_image_features.npy")
model = joblib.load("C:/Users/ashun/OneDrive/Desktop/test/lightgbm_fusion_multiclass_model.pkl")

# === Load Labels from CSV ===
df = pd.read_csv("C:/Users/ashun/OneDrive/Desktop/test/Review.csv")
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y_true = df['Sentiment'].str.lower().map(label_map).values

# === Combine Features ===
X = np.concatenate([text_feat, image_feat], axis=1)

# === Predict ===
y_pred = model.predict(X)

# === Save Labels and Predictions (optional) ===
#np.save("C:/Users/ashun/OneDrive/Desktop/test/y_test.npy", y_true)
#np.save("C:/Users/ashun/OneDrive/Desktop/test/y_pred.npy", y_pred)

# === Confusion Matrix ===
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
