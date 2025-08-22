package com.samsung.videotraffic.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.samsung.videotraffic.model.BatteryStats
import com.samsung.videotraffic.model.ClassificationResult
import com.samsung.videotraffic.model.TrafficStats

class MainViewModel : ViewModel() {
    
    private val _isMonitoring = MutableLiveData<Boolean>(false)
    val isMonitoring: LiveData<Boolean> = _isMonitoring
    
    private val _currentClassification = MutableLiveData<String>()
    val currentClassification: LiveData<String> = _currentClassification
    
    private val _networkStats = MutableLiveData<String>()
    val networkStats: LiveData<String> = _networkStats
    
    private val _batteryStats = MutableLiveData<BatteryStats>()
    val batteryStats: LiveData<BatteryStats> = _batteryStats
    
    private val _confidence = MutableLiveData<Float>(0f)
    val confidence: LiveData<Float> = _confidence
    
    fun startMonitoring() {
        _isMonitoring.value = true
    }
    
    fun stopMonitoring() {
        _isMonitoring.value = false
    }
    
    fun updateClassification(result: ClassificationResult) {
        val classificationText = when (result.classification) {
            ClassificationResult.Classification.REEL -> "REEL TRAFFIC DETECTED"
            ClassificationResult.Classification.NON_REEL -> "NON-REEL TRAFFIC DETECTED"
            ClassificationResult.Classification.UNKNOWN -> "ANALYZING TRAFFIC..."
        }
        _currentClassification.value = classificationText
        _confidence.value = result.confidence
    }
    
    fun updateNetworkStats(stats: TrafficStats) {
        val statsText = "Bytes: ${stats.bytesMonitored}, Packets: ${stats.packetsAnalyzed}"
        _networkStats.value = statsText
    }
    
    fun updateBatteryStats(stats: BatteryStats) {
        _batteryStats.value = stats
    }
    
    fun clearClassification() {
        _currentClassification.value = null
        _confidence.value = 0f
    }

    fun refreshData() {
        // Force refresh of all data
        _isMonitoring.value = _isMonitoring.value
        _currentClassification.value = _currentClassification.value
        _networkStats.value = _networkStats.value
        _batteryStats.value = _batteryStats.value
        _confidence.value = _confidence.value
    }
}
