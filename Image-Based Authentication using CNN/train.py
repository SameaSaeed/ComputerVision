import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_images_from_directory(directory, img_size=(64, 64)):
    images = []
    labels = []
    label_dict = {}
    label_count = 0

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            label_dict[label_count] = folder
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)  # Resize image to 64x64
                images.append(img)
                labels.append(label_count)
            label_count += 1

    images = np.array(images) / 255.0  # Normalize images to [0, 1]
    labels = np.array(labels)
    return images, labels, label_dict

# Load images from a dataset directory
images, labels, label_dict = load_images_from_directory('dataset')
print("Loaded images:", images.shape)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def create_cnn_model(input_shape=(64, 64, 3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer (softmax for classification)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_cnn_model(input_shape=(64, 64, 3), num_classes=len(label_dict))
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.2f}')

def authenticate_user(model, image_path, label_dict):
    # Preprocess the input image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Resize to the input size of the CNN
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and add batch dimension

    # Predict using the CNN
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Get the user's identity from the predicted class
    predicted_user = label_dict[predicted_class]
    print(f"Authenticated user: {predicted_user}")

# Example usage
authenticate_user(model, 'user_image.jpg', label_dict)