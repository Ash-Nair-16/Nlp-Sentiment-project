import numpy as np

text_features = np.load("C://Users//ashun//OneDrive//Desktop//test//sbert_text_features.npy")
print("Text Feature Shape:", text_features.shape)


image_features = np.load("C://Users//ashun//OneDrive//Desktop//test//resnet_image_features.npy")
print("Image Feature Shape:", image_features.shape)
