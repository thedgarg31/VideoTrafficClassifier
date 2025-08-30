# Implementation Details

- Real dataset collected across 10+ SNS platforms
- Network impairments simulated: congestion, jitter, throttling
- Feature engineering with entropy, burstiness, RTT stats
- Training pipeline: `train_reel_model.py`
- Automated deployment: `train_and_deploy.py` → builds TFLite + APK
