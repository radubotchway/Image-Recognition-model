import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import ssl

# Disable SSL verification to avoid SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define class names for CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display the first 16 images in a 4x4 grid
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

# Reduce the dataset size for faster computation (optional)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Load the pre-trained model
model = models.load_model('image_classifier.keras')

# Load and preprocess the image to be predicted
img = cv.imread('boat.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
img = cv.resize(img, (32, 32))  # Resize to 32x32 pixels
img = img / 255.0  # Normalize the image
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Display the preprocessed image
# plt.imshow(img[0], cmap=plt.cm.binary)
# plt.show()

# Make a prediction
prediction = model.predict(img)
index = np.argmax(prediction)
print(f'Prediction is: {class_names[index]}')