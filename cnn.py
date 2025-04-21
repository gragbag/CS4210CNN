#-------------------------------------------------------------------------
# AUTHOR: Julian Fong
# FILENAME: cnn.py
# SPECIFICATION: Use a CNN to create a model that identifies handwritten digits.
# FOR: CS 4210 - Assignment #4
# TIME SPENT: 1 hour
#-------------------------------------------------------------------------

# Importing Python libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        # Getting the label of the image (it's the first number in the filename)
        label = int(filename[0])

        # Converting images to grayscale and resizing them to 32x32
        img = Image.open(os.path.join(folder_path, filename)).convert('L').resize(image_size)

        # Adding the converted image to the feature matrix and label to the class vector
        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)

# Set your own paths here (relative to your project folder)
train_path = os.path.join("images", "train")
test_path = os.path.join("images", "test")

# Loading the raw images using the provided function. Hint: Use the provided load_digit_images_from_folder function that outputs X_train, Y_train for train_path and
# as X_test, Y_test for test_path
X_train, Y_train = load_digit_images_from_folder('train')
X_test, Y_test = load_digit_images_from_folder('test')

# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]. Hint: divide them by 255
X_train = X_train / 255
X_test = X_test / 255

# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
X_train = X_train.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

# Building a CNN model
model = models.Sequential([
    # Add a convolutional layer with 32 filters of size 3x3, relu activation, and input shape 32x32x1
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 1)),

    # Add a max pooling layer with pool size 2x2
    layers.MaxPool2D(pool_size=(2, 2)),

    # Add a flatten layer to convert the feature maps into a 1D vector
    layers.Flatten(),

    # Add a dense (fully connected) layer with 64 neurons and relu activation
    layers.Dense(64, activation='relu'),

    # Add the output layer with 10 neurons (digits 0–9) and softmax activation
    layers.Dense(10, activation='softmax')
])

# Compiling the model using optimizer = sgd, loss = sparse_categorical_crossentropy, and metric = accuracy
model.compile(optimizer= 'sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the model with batch_size=32 and epochs=10
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

# Evaluating the model on the test set
loss, acc = model.evaluate(X_test, Y_test)

# Printing the test accuracy
print("Accuracy: ", acc)
