import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import joblib
import os
import requests
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === OMDb API Key ===
OMDB_API_KEY = "655923f"  # your key

# === Load Models ===
model = joblib.load('lightgbm_fusion_multiclass_model.pkl')
sbert = SentenceTransformer('all-MiniLM-L6-v2')
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Class Mapping ===
class_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# === Utility Functions ===
def extract_text_features(text):
    return sbert.encode([text])[0]

def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    return features.squeeze().numpy()

def extract_movie_name(image_path):
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    return name.replace('_', ' ')

def fetch_movie_title(query_name):
    url = f"http://www.omdbapi.com/?t={query_name}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("Response") == "True":
            return data["Title"]
        else:
            return query_name
    except Exception as e:
        return f"Error: {e}"

def browse_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    image_path_var.set(path)
    if path:
        img = Image.open(path)
        img = img.resize((150, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        raw_name = extract_movie_name(path)
        official_title = fetch_movie_title(raw_name)
        movie_title_var.set(f"🎬 Movie: {official_title}")

def predict():
    text = text_entry.get("1.0", tk.END).strip()
    if not text or not image_path_var.get():
        messagebox.showwarning("Missing Input", "Please enter review text and select an image.")
        return

    try:
        text_feat = extract_text_features(text)
        image_feat = extract_image_features(image_path_var.get())
        fused = np.concatenate([text_feat, image_feat]).reshape(1, -1)

        probas = model.predict_proba(fused)[0]
        pred_class = np.argmax(probas)
        result_var.set(f"Predicted Sentiment: {class_map[pred_class]}")

        # Chart
        ax.clear()
        ax.bar(class_map.values(), probas, color=["red", "orange", "green"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        ax.set_title("Sentiment Probabilities")
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# === GUI Setup ===
root = tk.Tk()
root.title("🎬 Multimodal Sentiment Analyzer")
root.geometry("620x720")
root.configure(bg="#f4f4f4")

tk.Label(root, text="Multimodal Sentiment Analyzer", font=("Helvetica", 18, "bold"), bg="#f4f4f4", fg="#222").pack(pady=10)
tk.Label(root, text="Enter Review:", font=("Arial", 12), bg="#f4f4f4").pack()
text_entry = tk.Text(root, height=5, width=60)
text_entry.pack(pady=5)

tk.Button(root, text="📁 Select Poster Image", command=browse_image, bg="#007acc", fg="white", font=("Arial", 10), padx=10, pady=5).pack(pady=5)

image_label = tk.Label(root, bg="#f4f4f4")
image_label.pack()

image_path_var = tk.StringVar()
movie_title_var = tk.StringVar()

tk.Label(root, textvariable=image_path_var, font=("Arial", 10), bg="#f4f4f4", fg="#555").pack()
tk.Label(root, textvariable=movie_title_var, font=("Arial", 13, "italic"), bg="#f4f4f4", fg="#333").pack(pady=3)

tk.Button(root, text=" Predict Sentiment", command=predict, bg="#28a745", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5).pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Arial", 14, "bold"), fg="#222", bg="#f4f4f4").pack(pady=10)

fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()
