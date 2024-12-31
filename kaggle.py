import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load MNIST dataset (already split into train and test)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape to (height, width, channels) format
train_images = np.expand_dims(train_images, -1)  # Shape: (60000, 28, 28, 1)
test_images = np.expand_dims(test_images, -1)    # Shape: (10000, 28, 28, 1)

# Create ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Convert the images into a format that the model can train on
train_data = train_datagen.flow(train_images, train_labels, batch_size=32)
test_data = test_datagen.flow(test_images, test_labels, batch_size=32)

# Check if data is loading correctly
print(f"Training data: {train_data.samples} samples")
print(f"Test data: {test_data.samples} samples")
