# 🚀 Samsung EnnovateX 2025 - Build Instructions

## 📱 Android APK Build Guide

### Prerequisites
- Android Studio (latest version)
- Android SDK (API 24+)
- Java 17 or higher
- Git

### Quick Build Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/thedgarg31/VideoTrafficClassifier.git
   cd VideoTrafficClassifier
   ```

2. **Open in Android Studio**
   - Open Android Studio
   - Select "Open an existing project"
   - Navigate to the cloned directory and select it
   - Wait for Gradle sync to complete

3. **Build APK**
   - Go to `Build` → `Build Bundle(s) / APK(s)` → `Build APK(s)`
   - Or use terminal: `./gradlew assembleDebug`

4. **Find the APK**
   - The APK will be located at: `app/build/outputs/apk/debug/app-debug.apk`

### Alternative: Command Line Build

If you encounter file permission issues on Windows:

1. **Stop Gradle Daemon**
   ```bash
   ./gradlew --stop
   ```

2. **Clean Build Directory**
   ```bash
   # Windows PowerShell
   Remove-Item -Recurse -Force app\build -ErrorAction SilentlyContinue
   
   # Linux/Mac
   rm -rf app/build
   ```

3. **Build APK**
   ```bash
   ./gradlew assembleDebug
   ```

### Troubleshooting

#### File Permission Issues
If you get "Unable to delete directory" errors:
1. Close Android Studio completely
2. Stop all Gradle processes
3. Restart your computer
4. Try building again

#### Missing Resources
All resource issues have been fixed in the latest commit:
- ✅ Missing colors added
- ✅ Invalid attributes removed
- ✅ Material3 CardView styles fixed

#### TensorFlow Lite Warnings
The namespace warnings are normal and don't affect functionality.

### 🎯 What You Get

After successful build, you'll have:
- **Real-time reel traffic detection**
- **Advanced ML classification**
- **Privacy-compliant monitoring**
- **Professional Samsung-themed UI**
- **Central data collection capability**

### 📊 Testing the App

1. Install the APK on your Android device
2. Grant necessary permissions
3. Start monitoring traffic
4. View real-time classification results
5. Check data history and analytics

### 🔧 Advanced Features

- **ML Model Training**: Run `python3 train_reel_model.py`
- **End-to-End Pipeline**: Run `python3 train_and_deploy.py`
- **Central Data Collection**: Follow `CENTRAL_DATA_COLLECTION.md`

---

## 🏆 Samsung EnnovateX 2025 Compliance

✅ **Open Source**: All libraries are OSI-approved  
✅ **Privacy Compliant**: Metadata-only inspection  
✅ **Real-time Detection**: Sub-100ms inference  
✅ **Network Robust**: Handles congestion/jitter  
✅ **Mobile Optimized**: TensorFlow Lite integration  

---

**Repository**: https://github.com/thedgarg31/VideoTrafficClassifier.git  
**Documentation**: See README.md for complete project overview
