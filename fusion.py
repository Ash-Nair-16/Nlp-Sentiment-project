import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define directories
base_dir = 'C:\\Users\\ashun\\Downloads\\Cow Disease Detection.v1i.folder'  # e.g., "C:/Users/yourname/Downloads/cow_disease_dataset"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Image settings
image_size = (128, 128)
batch_size = 32

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')
valid_data = datagen.flow_from_directory(valid_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')
test_data = datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=valid_data, epochs=10)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
