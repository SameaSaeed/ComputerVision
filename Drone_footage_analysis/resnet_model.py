import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import numpy as np

class ResNetDroneClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_resnet_model(self, trainable_layers=False):
        """Build ResNet50-based model for drone image classification"""
        
        # Load pre-trained ResNet50 (without top classification layer)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = trainable_layers
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print(f"ResNet50 model built with {len(model.layers)} layers")
        print(f"Base model trainable: {trainable_layers}")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the ResNet model"""
        
        if self.model is None:
            raise ValueError("Model not built yet. Call build_resnet_model() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("ResNet model compiled successfully!")
        return self.model
    
    def fine_tune_model(self, learning_rate=0.0001):
        """Enable fine-tuning of the pre-trained layers"""
        
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Use a lower learning rate for fine-tuning
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("Model prepared for fine-tuning!")
        print(f"Fine-tuning learning rate: {learning_rate}")
    
    def get_feature_extractor(self):
        """Get feature extraction model (without classification head)"""
        
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        # Create feature extractor (base model + global average pooling)
        feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[1].output  # GlobalAveragePooling2D output
        )
        
        return feature_extractor

if __name__ == "__main__":
    # Create and build ResNet model
    resnet_classifier = ResNetDroneClassifier()
    model = resnet_classifier.build_resnet_model()
    resnet_classifier.compile_model()
    
    # Display model summary
    print("ResNet50-based Model Architecture:")
    print(f"Total parameters: {model.count_params():,}")
    
    # Show layer information
    print("\nModel layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} - {layer.__class__.__name__}")