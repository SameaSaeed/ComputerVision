import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class DroneImagePreprocessor:
    def __init__(self, data_dir='data', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_images_and_labels(self):
        """Load all images and their corresponding labels"""
        images = []
        labels = []
        
        print("Loading images from dataset...")
        
        for category in os.listdir(self.data_dir):
            category_path = os.path.join(self.data_dir, category)
            
            if not os.path.isdir(category_path):
                continue
                
            print(f"Processing category: {category}")
            
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, filename)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = self.preprocess_image(img)
                        images.append(img)
                        labels.append(category)
        
        print(f"Loaded {len(images)} images from {len(set(labels))} categories")
        return np.array(images), np.array(labels)
    
    def preprocess_image(self, img):
        """Preprocess individual image"""
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def prepare_dataset(self, test_size=0.2, val_size=0.1):
        """Prepare complete dataset with train/validation/test splits"""
        
        # Load images and labels
        images, labels = self.load_images_and_labels()
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels_encoded
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Dataset splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Testing: {len(X_test)} samples")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    preprocessor = DroneImagePreprocessor()
    train_data, val_data, test_data = preprocessor.prepare_dataset()
    
    # Save preprocessed data
    np.save('data/X_train.npy', train_data[0])
    np.save('data/y_train.npy', train_data[1])
    np.save('data/X_val.npy', val_data[0])
    np.save('data/y_val.npy', val_data[1])
    np.save('data/X_test.npy', test_data[0])
    np.save('data/y_test.npy', test_data[1])
    
    print("Preprocessed data saved successfully!")