# 📱 Mobile App Troubleshooting Guide

## 🚨 **App Crashing When Starting Classification**

### **Root Causes & Solutions:**

### 1. **📱 Permissions Issues**
**Problem**: App crashes when you tap "Start Classification"
**Solution**:
- When you first tap "Start Classification", Android will ask for permissions
- **Grant ALL permissions** when prompted:
  - ✅ Network State Access
  - ✅ WiFi State Access  
  - ✅ Internet Access
  - ✅ Notifications (Android 13+)

**Manual Permission Check**:
1. Go to **Settings → Apps → Video Traffic Classifier**
2. Tap **Permissions**
3. Enable ALL permissions listed

### 2. **🔋 Battery Optimization Killing Service**
**Problem**: App stops working when you leave it
**Solution**:
1. Go to **Settings → Battery → Battery Optimization**
2. Find **Video Traffic Classifier**
3. Select **"Don't optimize"** or **"Allow background activity"**

**Alternative path**:
- Settings → Apps → Video Traffic Classifier → Battery → Background Activity → **Allow**

### 3. **📊 Notification Permission (Android 13+)**
**Problem**: Service can't run without notification permission
**Solution**:
1. When app asks for notification permission → Tap **"Allow"**
2. Or manually: Settings → Apps → Video Traffic Classifier → Notifications → **Enable**

### 4. **🔐 READ_PHONE_STATE Permission**
**Problem**: Some devices require phone state access
**Solution**:
- If permission denied, app will still work with reduced accuracy
- Grant it manually if available: Settings → Apps → Permissions → Phone

---

## 📱 **Step-by-Step Testing Procedure**

### **Phase 1: Install & Setup**
1. Install the APK: `adb install app-debug.apk`
2. Open the app
3. **DON'T** tap Start Classification yet

### **Phase 2: Permissions**
1. Tap **"Start Classification"**
2. Android will show permission dialogs
3. **Tap "Allow" for EVERY permission request**
4. If no permissions appear, check Settings manually

### **Phase 3: Background Settings**
1. Go to phone Settings
2. Find **Video Traffic Classifier** in Apps list
3. Check these settings:
   - ✅ **Background App Refresh**: ON
   - ✅ **Battery Optimization**: OFF (don't optimize)
   - ✅ **Autostart**: ON (if available)
   - ✅ **Notifications**: ON

### **Phase 4: Test Classification**
1. Return to the app
2. Tap **"Start Classification"**
3. You should see:
   - Status: "Monitoring network traffic..."
   - A persistent notification appears
   - Button changes to "Stop Classification"

### **Phase 5: Generate Traffic**
1. **Keep the app open** initially
2. Open **YouTube** and play a video
3. Watch the app - should show "Video Traffic Detected"
4. Open **Chrome** and browse websites
5. Should show "Non-Video Traffic"

### **Phase 6: Background Test**
1. Press **Home button** (don't close app)
2. Use other apps (Instagram Reels, TikTok)
3. Return to Video Traffic Classifier
4. Check if monitoring is still active

---

## 🔧 **Device-Specific Issues**

### **Samsung Devices**
- **Samsung Battery Optimization**: Settings → Device Care → Battery → App Power Management → Apps that won't be put to sleep → Add Video Traffic Classifier

### **Xiaomi/MIUI Devices**
- **MIUI Optimization**: Settings → Apps → Manage Apps → Video Traffic Classifier → Other Permissions → Display Pop-up windows while running in background → **Enable**
- **Autostart**: Settings → Apps → Permissions → Autostart → Video Traffic Classifier → **Enable**

### **OnePlus/OxygenOS**
- **Battery Optimization**: Settings → Battery → Battery Optimization → All Apps → Video Traffic Classifier → Don't Optimize

### **Huawei Devices**
- **Protected Apps**: Settings → Apps → Advanced → Protected Apps → Video Traffic Classifier → **Enable**

---

## 🐛 **Common Error Patterns**

### **App Opens Then Immediately Closes**
```
Cause: Permission denied or security restriction
Fix: Check notification permission (Android 13+)
    Grant all permissions manually in Settings
```

### **"Start Classification" Button Does Nothing**
```
Cause: Service failed to start
Fix: Check background app restrictions
    Disable battery optimization
    Enable autostart if available
```

### **Classification Stops After Few Minutes**
```
Cause: Android killing background service
Fix: Add app to battery optimization whitelist
    Enable "Allow background activity"
```

### **No Traffic Detected**
```
Cause: No network activity or permission issues
Fix: Generate heavy traffic (YouTube videos)
    Check network permissions
    Restart monitoring
```

---

## ✅ **Success Indicators**

You'll know it's working when:
1. **Persistent notification** shows "Video Traffic Classifier - Monitoring network traffic..."
2. **App UI shows**: "Status: Monitoring network traffic..."
3. **Statistics update**: Bytes and packets counters increase
4. **Real-time classification**: Changes based on your activity
5. **Background survival**: Works even when you use other apps

---

## 📞 **Still Having Issues?**

### **Debug Steps:**
1. **Restart phone** after installing
2. **Clear app data**: Settings → Apps → Video Traffic Classifier → Storage → Clear Data
3. **Reinstall app** with fresh permissions
4. **Test on different Android device** if available

### **Device Requirements:**
- ✅ **Android 7.0+** (API 24+)
- ✅ **4GB+ RAM** (recommended)
- ✅ **Active internet connection**
- ✅ **Modern device** (2018+)

### **Alternative Testing:**
If problems persist, try this simplified test:
1. Install app
2. Grant ALL permissions immediately
3. Add to battery optimization whitelist BEFORE starting
4. Then test classification

The app should work reliably on most modern Android devices with proper permissions! 🚀
