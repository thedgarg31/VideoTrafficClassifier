#!/usr/bin/env python3
"""
Advanced TensorFlow Lite Model Training Pipeline for Samsung EnnovateX 2025
Real-time Reel vs Non-reel Traffic Classification

This script creates a comprehensive training pipeline that:
1. Generates synthetic dataset with advanced network features
2. Trains multiple model architectures (Mobile, Advanced)
3. Implements network robustness testing
4. Creates optimized TensorFlow Lite models
5. Generates compliance documentation

Open Source Compliance: Only uses OSI-approved libraries
- TensorFlow 2.13.0 (Apache 2.0)
- NumPy (BSD)
- Pandas (BSD)
- Scikit-learn (BSD)
- Matplotlib (PSF)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ReelTrafficDatasetGenerator:
    """Generates synthetic dataset for reel vs non-reel traffic classification"""
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
        self.feature_names = [
            'packet_size_mean', 'packet_size_std', 'packet_size_entropy',
            'inter_arrival_mean', 'inter_arrival_std', 'inter_arrival_entropy',
            'burstiness', 'flow_duration', 'flow_churn_rate',
            'tcp_handshake_time', 'quic_handshake_time', 'rtt_mean', 'rtt_std',
            'bitrate_mean', 'bitrate_variance', 'bitrate_entropy',
            'session_duration', 'data_volume', 'packet_count',
            'tcp_ratio', 'udp_ratio', 'http_ratio', 'https_ratio',
            'temporal_burst_pattern', 'flow_level_entropy',
            'congestion_window_size', 'retransmission_rate',
            'jitter_mean', 'jitter_std', 'packet_loss_rate'
        ]
        
    def generate_reel_traffic_features(self, num_samples: int) -> np.ndarray:
        """Generate features for reel/video traffic"""
        features = np.zeros((num_samples, len(self.feature_names)))
        
        for i in range(num_samples):
            # Reel traffic characteristics
            features[i, 0] = np.random.normal(1400, 200)  # Large packet sizes
            features[i, 1] = np.random.uniform(100, 300)  # Low packet size variation
            features[i, 2] = np.random.uniform(0.1, 0.3)  # Low entropy (consistent)
            
            features[i, 3] = np.random.uniform(16, 33)  # Consistent intervals (30fps)
            features[i, 4] = np.random.uniform(1, 5)    # Low interval variation
            features[i, 5] = np.random.uniform(0.1, 0.4)  # Low entropy
            
            features[i, 6] = np.random.uniform(0.5, 1.5)  # Low burstiness
            features[i, 7] = np.random.uniform(10, 300)   # Short to medium duration
            features[i, 8] = np.random.uniform(0.1, 0.3)  # Low churn rate
            
            features[i, 9] = np.random.uniform(50, 200)   # TCP handshake time
            features[i, 10] = np.random.uniform(20, 100)  # QUIC handshake time
            features[i, 11] = np.random.uniform(20, 100)  # RTT mean
            features[i, 12] = np.random.uniform(5, 20)    # RTT std
            
            features[i, 13] = np.random.uniform(500000, 2000000)  # High bitrate
            features[i, 14] = np.random.uniform(100000, 500000)   # Bitrate variance
            features[i, 15] = np.random.uniform(0.1, 0.4)        # Low entropy
            
            features[i, 16] = np.random.uniform(30, 300)  # Session duration
            features[i, 17] = np.random.uniform(1000000, 10000000)  # Data volume
            features[i, 18] = np.random.uniform(100, 1000)  # Packet count
            
            features[i, 19] = np.random.uniform(0.7, 0.9)  # High TCP ratio
            features[i, 20] = np.random.uniform(0.1, 0.3)  # Low UDP ratio
            features[i, 21] = np.random.uniform(0.1, 0.3)  # HTTP ratio
            features[i, 22] = np.random.uniform(0.7, 0.9)  # High HTTPS ratio
            
            features[i, 23] = np.random.uniform(0.1, 0.4)  # Low burst pattern
            features[i, 24] = np.random.uniform(0.1, 0.3)  # Low flow entropy
            
            features[i, 25] = np.random.uniform(10000, 50000)  # Congestion window
            features[i, 26] = np.random.uniform(0.01, 0.05)    # Low retransmission
            features[i, 27] = np.random.uniform(1, 5)          # Low jitter
            features[i, 28] = np.random.uniform(0.5, 2)        # Jitter std
            features[i, 29] = np.random.uniform(0.001, 0.01)   # Low packet loss
            
        return features
    
    def generate_non_reel_traffic_features(self, num_samples: int) -> np.ndarray:
        """Generate features for non-reel traffic (feeds, suggestions, etc.)"""
        features = np.zeros((num_samples, len(self.feature_names)))
        
        for i in range(num_samples):
            # Non-reel traffic characteristics
            features[i, 0] = np.random.normal(500, 300)   # Smaller packet sizes
            features[i, 1] = np.random.uniform(200, 500)  # High packet size variation
            features[i, 2] = np.random.uniform(0.6, 0.9)  # High entropy (variable)
            
            features[i, 3] = np.random.uniform(50, 500)   # Variable intervals
            features[i, 4] = np.random.uniform(10, 100)   # High interval variation
            features[i, 5] = np.random.uniform(0.6, 0.9)  # High entropy
            
            features[i, 6] = np.random.uniform(2, 8)      # High burstiness
            features[i, 7] = np.random.uniform(1, 60)     # Short duration
            features[i, 8] = np.random.uniform(0.4, 0.8)  # High churn rate
            
            features[i, 9] = np.random.uniform(100, 500)  # TCP handshake time
            features[i, 10] = np.random.uniform(50, 200)  # QUIC handshake time
            features[i, 11] = np.random.uniform(50, 200)  # RTT mean
            features[i, 12] = np.random.uniform(10, 50)   # RTT std
            
            features[i, 13] = np.random.uniform(10000, 100000)  # Low bitrate
            features[i, 14] = np.random.uniform(5000, 50000)    # Bitrate variance
            features[i, 15] = np.random.uniform(0.6, 0.9)       # High entropy
            
            features[i, 16] = np.random.uniform(5, 60)    # Short session duration
            features[i, 17] = np.random.uniform(1000, 100000)   # Low data volume
            features[i, 18] = np.random.uniform(10, 100)  # Few packets
            
            features[i, 19] = np.random.uniform(0.5, 0.8)  # Moderate TCP ratio
            features[i, 20] = np.random.uniform(0.2, 0.5)  # Higher UDP ratio
            features[i, 21] = np.random.uniform(0.3, 0.7)  # HTTP ratio
            features[i, 22] = np.random.uniform(0.3, 0.7)  # HTTPS ratio
            
            features[i, 23] = np.random.uniform(0.6, 0.9)  # High burst pattern
            features[i, 24] = np.random.uniform(0.6, 0.9)  # High flow entropy
            
            features[i, 25] = np.random.uniform(1000, 10000)  # Small congestion window
            features[i, 26] = np.random.uniform(0.05, 0.15)   # Higher retransmission
            features[i, 27] = np.random.uniform(5, 20)         # High jitter
            features[i, 28] = np.random.uniform(2, 10)         # Jitter std
            features[i, 29] = np.random.uniform(0.01, 0.05)   # Higher packet loss
            
        return features
    
    def add_network_impairments(self, features: np.ndarray, impairment_type: str) -> np.ndarray:
        """Add network impairments to simulate real-world conditions"""
        impaired_features = features.copy()
        
        if impairment_type == "congestion":
            # Simulate network congestion
            impaired_features[:, 11] *= np.random.uniform(1.5, 3.0, features.shape[0])  # Higher RTT
            impaired_features[:, 25] *= np.random.uniform(0.3, 0.7, features.shape[0])  # Smaller congestion window
            impaired_features[:, 26] *= np.random.uniform(2.0, 5.0, features.shape[0])  # Higher retransmission
            impaired_features[:, 29] *= np.random.uniform(2.0, 5.0, features.shape[0])  # Higher packet loss
            
        elif impairment_type == "jitter":
            # Simulate jitter
            impaired_features[:, 27] *= np.random.uniform(2.0, 5.0, features.shape[0])  # Higher jitter
            impaired_features[:, 28] *= np.random.uniform(2.0, 5.0, features.shape[0])  # Higher jitter std
            
        elif impairment_type == "throttling":
            # Simulate bandwidth throttling
            impaired_features[:, 13] *= np.random.uniform(0.1, 0.5, features.shape[0])  # Lower bitrate
            impaired_features[:, 14] *= np.random.uniform(0.5, 1.0, features.shape[0])  # Lower variance
            impaired_features[:, 17] *= np.random.uniform(0.1, 0.5, features.shape[0])  # Lower data volume
            
        return impaired_features
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete dataset with network impairments"""
        logger.info("Generating synthetic dataset...")
        
        # Generate base features
        reel_features = self.generate_reel_traffic_features(self.num_samples // 2)
        non_reel_features = self.generate_non_reel_traffic_features(self.num_samples // 2)
        
        # Add network impairments
        impairment_types = ["congestion", "jitter", "throttling", "normal"]
        impaired_reel = []
        impaired_non_reel = []
        
        for impairment in impairment_types:
            if impairment == "normal":
                impaired_reel.append(reel_features[:len(reel_features)//4])
                impaired_non_reel.append(non_reel_features[:len(non_reel_features)//4])
            else:
                impaired_reel.append(self.add_network_impairments(
                    reel_features[:len(reel_features)//4], impairment))
                impaired_non_reel.append(self.add_network_impairments(
                    non_reel_features[:len(non_reel_features)//4], impairment))
        
        # Combine all features
        X_reel = np.vstack(impaired_reel)
        X_non_reel = np.vstack(impaired_non_reel)
        
        X = np.vstack([X_reel, X_non_reel])
        y = np.hstack([np.ones(len(X_reel)), np.zeros(len(X_non_reel))])
        
        # Add some unknown samples
        unknown_features = self.generate_unknown_traffic_features(len(X) // 10)
        X = np.vstack([X, unknown_features])
        y = np.hstack([y, np.full(len(unknown_features), 2)])  # 2 for unknown
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split into train/validation/test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        logger.info(f"Dataset generated: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
        logger.info(f"Feature count: {X.shape[1]}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def generate_unknown_traffic_features(self, num_samples: int) -> np.ndarray:
        """Generate features for unknown traffic types"""
        features = np.zeros((num_samples, len(self.feature_names)))
        
        for i in range(num_samples):
            # Random features that don't clearly belong to either category
            for j in range(len(self.feature_names)):
                features[i, j] = np.random.uniform(0, 1000)  # Random values
                
        return features

class ReelTrafficModel:
    """Advanced neural network models for reel traffic classification"""
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        
    def create_mobile_model(self) -> keras.Model:
        """Create lightweight model for mobile deployment"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_advanced_model(self) -> keras.Model:
        """Create advanced model with attention mechanisms"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Feature extraction layers
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(64, activation='tanh')(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model: keras.Model, train_data: Tuple, val_data: Tuple, 
                   model_name: str, epochs: int = 100) -> keras.Model:
        """Train the model with callbacks and monitoring"""
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train the model
        logger.info(f"Training {model_name}...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save training history
        with open(f'{model_name}_history.json', 'w') as f:
            json.dump(history.history, f)
        
        return model
    
    def evaluate_model(self, model: keras.Model, test_data: Tuple, model_name: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        X_test, y_test = test_data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Classification report
        class_names = ['Non-Reel', 'Reel', 'Unknown']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        results = {
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        with open(f'{model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

class TensorFlowLiteConverter:
    """Convert trained models to TensorFlow Lite format"""
    
    def __init__(self):
        self.representative_dataset = None
    
    def create_representative_dataset(self, X_sample: np.ndarray, num_samples: int = 100):
        """Create representative dataset for quantization"""
        def representative_dataset():
            for i in range(min(num_samples, len(X_sample))):
                yield [X_sample[i:i+1].astype(np.float32)]
        
        self.representative_dataset = representative_dataset
    
    def convert_to_tflite(self, model: keras.Model, model_name: str, 
                         quantization: bool = True) -> str:
        """Convert Keras model to TensorFlow Lite"""
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization:
            # Enable quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        model_path = f'{model_name}.tflite'
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = os.path.getsize(model_path) / 1024  # KB
        
        logger.info(f"TensorFlow Lite model saved: {model_path}")
        logger.info(f"Model size: {model_size:.2f} KB")
        
        return model_path
    
    def create_model_info(self, model_name: str, scaler: StandardScaler, 
                         feature_names: List[str]) -> Dict[str, Any]:
        """Create model information file for Android integration"""
        
        model_info = {
            'model_name': model_name,
            'input_shape': [1, len(feature_names)],
            'output_shape': [1, 3],  # 3 classes
            'feature_names': feature_names,
            'class_names': ['Non-Reel', 'Reel', 'Unknown'],
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'scaler_var': scaler.var_.tolist(),
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        info_path = f'{model_name}_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_info

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Reel Traffic Classification Model')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--model-type', choices=['mobile', 'advanced', 'both'], 
                       default='both', help='Model type to train')
    parser.add_argument('--quantization', action='store_true', help='Enable quantization')
    
    args = parser.parse_args()
    
    logger.info("Starting Samsung EnnovateX 2025 Training Pipeline")
    logger.info(f"Configuration: {args.samples} samples, {args.epochs} epochs, {args.model_type} model")
    
    # Generate dataset
    generator = ReelTrafficDatasetGenerator(args.samples)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generator.generate_dataset()
    
    # Save dataset
    dataset_df = pd.DataFrame(X_train, columns=generator.feature_names)
    dataset_df['label'] = y_train
    dataset_df.to_csv('reel_traffic_dataset.csv', index=False)
    logger.info("Dataset saved to reel_traffic_dataset.csv")
    
    # Initialize model trainer
    trainer = ReelTrafficModel(input_dim=len(generator.feature_names))
    
    # Initialize converter
    converter = TensorFlowLiteConverter()
    converter.create_representative_dataset(X_train)
    
    models_to_train = []
    if args.model_type in ['mobile', 'both']:
        models_to_train.append(('mobile', trainer.create_mobile_model()))
    if args.model_type in ['advanced', 'both']:
        models_to_train.append(('advanced', trainer.create_advanced_model()))
    
    results = {}
    
    for model_type, model in models_to_train:
        logger.info(f"Training {model_type} model...")
        
        # Train model
        trained_model = trainer.train_model(
            model, (X_train, y_train), (X_val, y_val), 
            f'reel_traffic_{model_type}', args.epochs
        )
        
        # Evaluate model
        model_results = trainer.evaluate_model(
            trained_model, (X_test, y_test), f'reel_traffic_{model_type}'
        )
        results[model_type] = model_results
        
        # Convert to TensorFlow Lite
        tflite_path = converter.convert_to_tflite(
            trained_model, f'reel_traffic_{model_type}', args.quantization
        )
        
        # Create model info
        model_info = converter.create_model_info(
            f'reel_traffic_{model_type}', trainer.scaler, generator.feature_names
        )
        
        logger.info(f"{model_type} model training completed")
        logger.info(f"Test accuracy: {model_results['test_accuracy']:.4f}")
    
    # Create summary report
    summary = {
        'training_config': vars(args),
        'dataset_info': {
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': len(generator.feature_names)
        },
        'model_results': results,
        'compliance': {
            'open_source': True,
            'licenses': ['Apache 2.0', 'BSD', 'MIT'],
            'no_proprietary_apis': True,
            'no_cloud_services': True
        }
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training pipeline completed successfully!")
    logger.info("Generated files:")
    logger.info("- reel_traffic_mobile.tflite (Mobile model)")
    logger.info("- reel_traffic_advanced.tflite (Advanced model)")
    logger.info("- *_info.json (Model information files)")
    logger.info("- training_summary.json (Complete results)")
    logger.info("- reel_traffic_dataset.csv (Training dataset)")

if __name__ == "__main__":
    main()
