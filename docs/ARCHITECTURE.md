# Technical Architecture

## Pipeline
1. **Traffic Capture** → Real-time packet metadata collection
2. **Feature Extraction** → 30+ features (packet size, inter-arrival, RTT, burstiness)
3. **Model Inference** → TFLite model (<100ms latency)
4. **Fallback Heuristics** → Rule-based classification if ML confidence < threshold
5. **UI Display** → Live feedback in Android app

## Deployment
- Mobile-optimized models (<100KB)
- Android integration with `ReelTrafficClassifier.kt`
- Real-time monitoring with `AdvancedTrafficMonitorService.kt`
