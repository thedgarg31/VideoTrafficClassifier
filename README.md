# Video Traffic Classifier - Samsung EnnovateX 2025

**Samsung EnnovateX 2025 AI Challenge - Problem Statement #9**  
Real-time AI-powered classification of video vs non-video network traffic for Android devices.

## 🎯 Project Overview

This Android application implements a real-time machine learning solution to classify network traffic as either **video** or **non-video** content. The app monitors device network activity and uses a lightweight TensorFlow Lite model to provide instant classifications.

### Key Features

- ✅ **Real-time Traffic Classification**: Monitors and classifies network traffic in real-time
- ✅ **TensorFlow Lite Integration**: Uses optimized ML models for mobile devices
- ✅ **Heuristic Fallback**: Intelligent rule-based classification when ML model unavailable
- ✅ **Beautiful UI**: Modern Material Design interface with Samsung branding
- ✅ **Background Monitoring**: Foreground service for continuous operation
- ✅ **Traffic Statistics**: Real-time display of bytes monitored and packets analyzed
- ✅ **Permission Management**: Handles network monitoring permissions gracefully

## 📱 Screenshots & Demo

The app provides a clean, intuitive interface showing:
- Current monitoring status
- Real-time classification results (Video/Non-Video)
- Confidence percentages
- Traffic statistics (bytes monitored, packets analyzed)
- Samsung EnnovateX 2025 branding

## 🚀 Quick Start

### Prerequisites

- Android Studio 2023.1.1 (Hedgehog) or later
- Java Development Kit (JDK) 17 or higher (compatible with Java 21)
- Android SDK 34
- Minimum Android version: 7.0 (API level 24)
- Gradle 8.12
- Kotlin 1.9.20+

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/[your-username]/VideoTrafficClassifier.git
   cd VideoTrafficClassifier
   ```

2. **Open in Android Studio**
   - Launch Android Studio
   - Click "Open an Existing Project"
   - Navigate to the cloned directory
   - Wait for Gradle sync to complete

3. **Build the Project**
   ```bash
   ./gradlew assembleDebug
   ```

4. **Install on Device**
   ```bash
   ./gradlew installDebug
   ```

### Building APK for Distribution

```bash
# Debug APK
./gradlew assembleDebug

# Release APK (requires signing)
./gradlew assembleRelease
```

The APK will be generated in `app/build/outputs/apk/`

## 🧠 AI/ML Model Architecture

### Current Implementation

The app supports two classification approaches:

1. **TensorFlow Lite Model** (Primary)
   - Input: 10 network traffic features
   - Output: Binary classification (video/non-video) with confidence
   - Model file: `video_traffic_model.tflite`

2. **Heuristic Classifier** (Fallback)
   - Rule-based classification using traffic patterns
   - Considers bitrate, packet size, timing, and data volume
   - Provides reliable baseline performance

### Feature Engineering

The classifier analyzes these network traffic features:

- **Packet Size**: Average size of network packets
- **Bitrate**: Data transmission rate (bits per second)
- **Packet Interval**: Time between packets
- **Burstiness**: Variation in packet timing
- **Protocol Ratios**: TCP vs UDP usage
- **Packet Gap**: Average time between packets
- **Size Variation**: Consistency of packet sizes
- **Connection Duration**: How long the connection has been active
- **Data Volume**: Total amount of data transferred

### Training Your Own Model

To train a custom TensorFlow Lite model:

1. **Collect Training Data**
   ```python
   # Example data collection script
   import pandas as pd
   import numpy as np
   
   # Collect network traffic features
   # Label data as video (1) or non-video (0)
   features = ['packet_size', 'bitrate', 'packet_interval', ...]
   ```

2. **Train the Model**
   ```python
   import tensorflow as tf
   from tensorflow import keras
   
   # Create model
   model = keras.Sequential([
       keras.layers.Dense(64, activation='relu', input_shape=(10,)),
       keras.layers.Dropout(0.3),
       keras.layers.Dense(32, activation='relu'),
       keras.layers.Dense(2, activation='softmax')  # Binary classification
   ])
   
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   
   # Train model
   model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
   ```

3. **Convert to TensorFlow Lite**
   ```python
   # Convert the model
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   
   # Save the model
   with open('video_traffic_model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

4. **Replace Model in App**
   - Copy the new `.tflite` file to `app/src/main/assets/`
   - Rebuild and test the app

## 🔧 Configuration

### Monitoring Parameters

You can adjust monitoring behavior in `TrafficMonitorService.kt`:

```kotlin
companion object {
    private const val MONITORING_INTERVAL = 2000L // 2 seconds
    private const val PACKET_HISTORY_SIZE = 100   // Keep last 100 packets
}
```

### Classification Thresholds

Modify heuristic thresholds in `VideoTrafficClassifier.kt`:

```kotlin
// High bitrate threshold for video detection
if (features.bitrate > 1_000_000) { // 1 Mbps
    videoScore += 2f
}

// Video confidence threshold
val isVideo = confidence > 0.6f // 60% confidence
```

## 📱 APK Installation Guide

### Installing on Android Device

1. **Enable Developer Options**
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
   - Developer Options will appear in Settings

2. **Enable USB Debugging**
   - Go to Settings → Developer Options
   - Enable "USB Debugging"

3. **Install APK**
   ```bash
   # Via ADB
   adb install app/build/outputs/apk/debug/app-debug.apk
   
   # Or copy APK to device and install manually
   ```

4. **Grant Permissions**
   - Open the app
   - Grant network monitoring permissions when prompted
   - Start monitoring by tapping the button

### Installing via Android Studio

1. Connect your Android device via USB
2. Click the "Run" button in Android Studio
3. Select your device from the deployment target
4. The app will be installed and launched automatically

## 🛠️ Development

### Project Structure

```
app/
├── src/main/java/com/samsung/videotraffic/
│   ├── MainActivity.kt              # Main UI controller
│   ├── model/                       # Data models
│   │   ├── ClassificationResult.kt  # ML classification results
│   │   ├── TrafficFeatures.kt       # Network traffic features
│   │   └── TrafficStats.kt          # Statistics tracking
│   ├── ml/                          # Machine learning
│   │   └── VideoTrafficClassifier.kt # TensorFlow Lite classifier
│   └── service/                     # Background services
│       └── TrafficMonitorService.kt  # Network monitoring service
├── src/main/res/                    # Android resources
│   ├── layout/                      # UI layouts
│   ├── values/                      # Strings, colors, themes
│   └── drawable/                    # Icons and graphics
└── src/main/assets/                 # TensorFlow Lite model
    └── video_traffic_model.tflite
```

### Key Technologies

- **Kotlin**: Primary development language
- **TensorFlow Lite**: On-device machine learning
- **Android Jetpack**: Modern Android development components
- **Material Design**: UI/UX framework
- **Coroutines**: Asynchronous programming
- **LiveData**: Reactive UI updates

### Testing

```bash
# Run unit tests
./gradlew test

# Run instrumented tests
./gradlew connectedAndroidTest
```

## 🔒 Permissions & Privacy

The app requires these permissions for network monitoring:

- `INTERNET`: Access network connections
- `ACCESS_NETWORK_STATE`: Monitor network status
- `ACCESS_WIFI_STATE`: Monitor WiFi status
- `READ_PHONE_STATE`: Access device network info
- `FOREGROUND_SERVICE`: Run background monitoring

**Privacy Note**: The app only analyzes traffic patterns and does not inspect or store actual data content.

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Ensure `video_traffic_model.tflite` exists in assets
   - App will fallback to heuristic classification

2. **No Traffic Detected**
   - Generate network activity (browse web, stream video)
   - Check that permissions are granted
   - Restart the monitoring service

3. **Build Errors**
   - Clean and rebuild: `./gradlew clean build`
   - Check Android SDK installation
   - Verify Gradle version compatibility

4. **Permission Denied**
   - Manually grant permissions in Settings → Apps → Video Traffic Classifier
   - Some devices require additional security permissions

### Logs & Debugging

```bash
# View app logs
adb logcat | grep VideoTrafficClassifier

# View specific component logs
adb logcat | grep TrafficMonitorService
adb logcat | grep VideoTrafficClassifier
```

## 🎯 Performance Optimization

### Battery Optimization
- Monitoring runs every 2 seconds (configurable)
- Uses efficient system APIs for traffic stats
- Minimal CPU usage with coroutines

### Memory Management
- Keeps only last 100 packets in memory
- Automatic cleanup of old data
- Efficient TensorFlow Lite model loading

### Network Impact
- No additional network requests
- Only monitors existing device traffic
- Zero impact on user's data usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Samsung EnnovateX 2025

This project was developed for the **Samsung EnnovateX 2025 AI Challenge**, specifically addressing **Problem Statement #9**: Real-time classification of video vs non-video network traffic.

### Challenge Requirements Met

- ✅ Real-time traffic classification
- ✅ Android mobile application
- ✅ Machine learning integration
- ✅ User-friendly interface
- ✅ Efficient resource usage
- ✅ Open source implementation

## 📞 Support

For questions or support:
- Contact: [dgav3105@gmail.com]
- Samsung EnnovateX Challenge Portal

## 🔮 Future Enhancements

- [ ] Deep packet inspection for more accurate classification
- [ ] Support for additional traffic types (gaming, VoIP, etc.)
- [ ] Cloud-based model updates
- [ ] Advanced visualization and analytics
- [ ] Integration with network QoS systems
- [ ] Real-time bandwidth optimization recommendations

---

**Built with ❤️ for Samsung EnnovateX 2025**
