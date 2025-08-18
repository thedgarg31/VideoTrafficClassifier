#!/usr/bin/env python3
"""
Video Traffic Classification Model Training Script
Samsung EnnovateX 2025 AI Challenge - Problem Statement #9

This script trains a TensorFlow model to classify network traffic as video or non-video
and converts it to TensorFlow Lite format for mobile deployment.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data(num_samples=10000):
    """
    Generate synthetic network traffic data for training.
    In a real implementation, this would be replaced with actual traffic data.
    """
    np.random.seed(42)
    
    # Video traffic characteristics
    video_samples = num_samples // 2
    video_data = []
    
    for _ in range(video_samples):
        # Video traffic typically has:
        # - High bitrate (1-10 Mbps)
        # - Large packet sizes (800-1500 bytes)
        # - Regular intervals (streaming)
        # - Low burstiness
        # - High TCP ratio
        packet_size = np.random.normal(1200, 200)
        bitrate = np.random.normal(3_000_000, 1_000_000)  # 3 Mbps average
        packet_interval = np.random.normal(40, 10)  # ~25 FPS
        burstiness = np.random.normal(1.2, 0.3)
        tcp_ratio = np.random.normal(0.9, 0.1)
        udp_ratio = 1 - tcp_ratio
        avg_packet_gap = packet_interval + np.random.normal(0, 5)
        packet_size_variation = np.random.normal(100, 30)
        connection_duration = np.random.normal(120_000, 60_000)  # 2 minutes avg
        data_volume = bitrate * (connection_duration / 1000) / 8
        
        video_data.append([
            max(0, packet_size),
            max(0, bitrate),
            max(0, packet_interval),
            max(0, burstiness),
            np.clip(tcp_ratio, 0, 1),
            np.clip(udp_ratio, 0, 1),
            max(0, avg_packet_gap),
            max(0, packet_size_variation),
            max(0, connection_duration),
            max(0, data_volume),
            1  # Label: 1 = video
        ])
    
    # Non-video traffic characteristics
    non_video_samples = num_samples - video_samples
    non_video_data = []
    
    for _ in range(non_video_samples):
        # Non-video traffic typically has:
        # - Lower bitrate (10-500 kbps)
        # - Smaller packet sizes (64-800 bytes)
        # - Irregular intervals
        # - Higher burstiness
        # - Mixed protocols
        packet_size = np.random.normal(400, 200)
        bitrate = np.random.normal(200_000, 150_000)  # 200 kbps average
        packet_interval = np.random.exponential(100)  # More irregular
        burstiness = np.random.normal(3.0, 1.0)
        tcp_ratio = np.random.normal(0.7, 0.2)
        udp_ratio = 1 - tcp_ratio
        avg_packet_gap = packet_interval + np.random.normal(0, 20)
        packet_size_variation = np.random.normal(150, 50)
        connection_duration = np.random.normal(30_000, 20_000)  # 30 seconds avg
        data_volume = bitrate * (connection_duration / 1000) / 8
        
        non_video_data.append([
            max(0, packet_size),
            max(0, bitrate),
            max(0, packet_interval),
            max(0, burstiness),
            np.clip(tcp_ratio, 0, 1),
            np.clip(udp_ratio, 0, 1),
            max(0, avg_packet_gap),
            max(0, packet_size_variation),
            max(0, connection_duration),
            max(0, data_volume),
            0  # Label: 0 = non-video
        ])
    
    # Combine data
    all_data = video_data + non_video_data
    np.random.shuffle(all_data)
    
    columns = [
        'packet_size', 'bitrate', 'packet_interval', 'burstiness',
        'tcp_ratio', 'udp_ratio', 'avg_packet_gap', 'packet_size_variation',
        'connection_duration', 'data_volume', 'is_video'
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    return df

def create_model(input_shape):
    """Create a neural network model for traffic classification."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        
        keras.layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and show metrics."""
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Video', 'Video']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Video', 'Video'],
                yticklabels=['Non-Video', 'Video'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def convert_to_tflite(model, X_test):
    """Convert the trained model to TensorFlow Lite format."""
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use representative dataset for quantization (optional)
    def representative_data_gen():
        for input_value in X_test[:100]:  # Use first 100 test samples
            yield [input_value.reshape(1, -1).astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save the model
    with open('app/src/main/assets/video_traffic_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to app/src/main/assets/video_traffic_model.tflite")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model

def test_tflite_model(tflite_model, X_test, y_test):
    """Test the TensorFlow Lite model."""
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on a few samples
    correct = 0
    total = min(100, len(X_test))  # Test on first 100 samples
    
    for i in range(total):
        # Set input tensor
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        
        if predicted_class == y_test[i]:
            correct += 1
    
    accuracy = correct / total
    print(f"\nTensorFlow Lite model accuracy on {total} test samples: {accuracy:.4f}")

def main():
    """Main training pipeline."""
    print("Video Traffic Classification Model Training")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    df = generate_synthetic_data(num_samples=10000)
    
    # Display data info
    print(f"Dataset shape: {df.shape}")
    print(f"Video samples: {df['is_video'].sum()}")
    print(f"Non-video samples: {len(df) - df['is_video'].sum()}")
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col != 'is_video']
    X = df[feature_columns].values
    y = df['is_video'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_model(input_shape=X_train_scaled.shape[1])
    
    # Print model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    evaluate_model(model, X_test_scaled, y_test)
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    tflite_model = convert_to_tflite(model, X_test_scaled)
    
    # Test TFLite model
    test_tflite_model(tflite_model, X_test_scaled, y_test)
    
    # Save training info
    with open('model_info.txt', 'w') as f:
        f.write(f"Video Traffic Classification Model\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {feature_columns}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        f.write(f"Model size: {len(tflite_model) / 1024:.2f} KB\n")
    
    print("\nTraining completed successfully!")
    print("Files generated:")
    print("- app/src/main/assets/video_traffic_model.tflite")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- model_info.txt")

if __name__ == "__main__":
    main()
