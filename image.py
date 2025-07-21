# image_feature_extraction.py

import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# === RESNET IMAGE FEATURE EXTRACTOR FROM URL ===
class ImageFeatureExtractorFromURL:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # remove classifier
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_url: str) -> np.ndarray:
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                features = self.model(image)

            return features.squeeze().numpy().flatten()
        except Exception as e:
            print(f"❌ Error processing image URL {image_url}: {e}")
            return np.zeros(2048)  # fallback

# === MAIN PROCESS ===
def main():
    # Path to your CSV
    csv_path = 'C://Users//ashun//OneDrive//Desktop//test//Review.csv'
    df = pd.read_csv(csv_path)

    # Initialize extractor
    extractor = ImageFeatureExtractorFromURL()

    # Extract image features
    print("🖼️ Starting image feature extraction...")
    image_features = []
    for url in df['Poster']:
        features = extractor.extract(url)
        image_features.append(features)

    # Convert to array and save
    image_features = np.array(image_features)
    np.save('C://Users//ashun/OneDrive//Desktop//test//resnet_image_features.npy', image_features)

    print("✅ Done! Image features shape:", image_features.shape)

if __name__ == "__main__":
    main()
