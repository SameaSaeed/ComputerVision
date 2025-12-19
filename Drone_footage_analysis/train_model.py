import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from resnet_model import ResNetDroneClassifier
from cnn_model import DroneImageCNN

class ModelTrainer:
    def __init__(self, model_type='resnet'):
        self.model_type = model_type
        self.model = None
        self.history = None
        self.label_encoder = None
        
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def load_data(self):
        """Load preprocessed training data"""
        print("Loading preprocessed data...")
        
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_model(self):
        """Create model based on specified type"""
        
        if self.model_type == 'resnet':
            classifier = ResNetDroneClassifier(num_classes=len(self.label_encoder.classes_))
            model = classifier.build_resnet_model()
            classifier.compile_model(learning_rate=0.001)
            self.model = model
            self.classifier = classifier
            
        elif self.model_type == 'custom_cnn':
            classifier = DroneImageCNN(num_classes=len(self.label_encoder.classes_))
            model = classifier.build_custom_cnn()
            classifier.compile_model(learning_rate=0.001)
            self.model = model
            self.classifier = classifier
        
        else:
            raise ValueError("model_type must be 'resnet' or 'custom_cnn'")
        
        print(f"Created {self.model_type} model")
        return self.model
    
    def train_model(self, train_data, val_data, epochs=20, batch_size=32):
        """Train the model"""
        
        if self.model is None:
            self.create_model()
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'models/best_{self.model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"Starting training for {epochs} epochs...")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        X_test, y_test = test_data
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'predicted_classes': y_pred_classes
        }
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{self.model_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved as results/{self.model_type}_training_history.png")

def main():
    """Main training function"""
    
    # Choose model type
    model_type = 'resnet'  # Change to 'custom_cnn' to use custom CNN
    
    # Create trainer
    trainer = ModelTrainer(model_type=model_type)
    
    # Load data
    train_data, val_data, test_data = trainer.load_data()
    
    # Train model
    history = trainer.train_model(
        train_data, val_data,
        epochs=15,
        batch_size=16
    )
    
    # Evaluate model
    results = trainer.evaluate_model(test_data)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save results
    with open(f'results/{model_type}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nTraining and evaluation completed!")
    print(f"Model saved as: models/best_{model_type}_model.h5")
    print(f"Results saved as: results/{model_type}_results.pkl")

if __name__ == "__main__":
    main()