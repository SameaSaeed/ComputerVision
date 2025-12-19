import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import cv2
import os

class FeatureExtractor:
    def __init__(self, model_path='models/best_resnet_model.h5'):
        self.model_path = model_path
        self.model = None
        self.feature_extractor = None
        self.label_encoder = None
        
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        
        # Create feature extractor (remove classification layers)
        # For ResNet model, extract features from GlobalAveragePooling2D layer
        feature_layer_index = -4  # Adjust based on your model architecture
        
        self.feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[feature_layer_index].output
        )
        
        print("Feature extractor created successfully!")
        print(f"Feature vector size: {self.feature_extractor.output_shape[1]}")
        
        return self.feature_extractor
    
    def extract_features(self, images):
        """Extract features from images"""
        
        if self.feature_extractor is None:
            self.load_model()
        
        print(f"Extracting features from {len(images)} images...")
        features = self.feature_extractor.predict(images, batch_size=32, verbose=1)
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def visualize_features_pca(self, features, labels, save_path='results/features_pca.png'):
        """Visualize features using PCA"""
        
        print("Performing PCA analysis...")
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green', 'orange']
        class_names = self.label_encoder.classes_
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=colors[i % len(colors)],
                label=class_name,
                alpha=0.7,
                s=50
            )
        
        plt.xlabel(f'First Principal Component (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('Feature Visualization using PCA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"PCA visualization saved as {save_path}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        
        return features_2d, pca
    
    def visualize_features_tsne(self, features, labels, save_path='results/features_tsne.png'):
        """Visualize features using t-SNE"""
        
        print("Performing t-SNE analysis...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green', 'orange']
        class_names = self.label_encoder.classes_
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=colors[i % len(colors)],
                label=class_name,
                alpha=0.7,
                s=50
            )
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Feature Visualization using t-SNE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"t-SNE visualization saved as {save_path}")
        
        return features_2d, tsne            