import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os

# Load annotations from CSV
annotations_df = pd.read_csv(r'C:\Users\tejas\OneDrive\Documents\project\train.csv')

# Function to load and preprocess images
def load_and_preprocess_images(annotations_df):
    images = []
    labels = []

    for index, row in annotations_df.iterrows():
        image_path = os.path.join(
            r'C:\Users\tejas\OneDrive\Documents\project\train\cat',
            row['filename']
        )
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {row['filename']}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (224, 224))  # Resize images to a fixed size (adjust as needed)
        image = image / 255.0  # Normalize pixel values to be between 0 and 1
        images.append(image)

        # Extract bounding box coordinates and normalize them
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        labels.append([xmin / 224, ymin / 224, xmax / 224, ymax / 224])

    return np.array(images), np.array(labels)


# Load and preprocess the dataset
images, labels = load_and_preprocess_images(annotations_df)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='sigmoid')  # Output layer with 4 neurons for xmin, ymin, xmax, ymax
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
