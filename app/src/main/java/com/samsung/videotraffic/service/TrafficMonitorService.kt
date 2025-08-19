package com.samsung.videotraffic.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.net.TrafficStats
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.samsung.videotraffic.R
import com.samsung.videotraffic.ml.VideoTrafficClassifier
import com.samsung.videotraffic.model.ClassificationResult
import com.samsung.videotraffic.model.TrafficFeatures
import com.samsung.videotraffic.model.TrafficStats as AppTrafficStats
import kotlinx.coroutines.*
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

class TrafficMonitorService : Service() {

    private val binder = LocalBinder()
    
    private lateinit var classifier: VideoTrafficClassifier
    private var monitoringJob: Job? = null
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // LiveData for UI updates
    private val _classificationResult = MutableLiveData<ClassificationResult>()
    val classificationResult: LiveData<ClassificationResult> = _classificationResult
    
    private val _trafficStats = MutableLiveData<AppTrafficStats>()
    val trafficStats: LiveData<AppTrafficStats> = _trafficStats
    
    // Traffic monitoring variables
    private var startTime = 0L
    private var totalBytesMonitored = 0L
    private var packetsAnalyzed = 0
    private var lastMeasurementTime = 0L
    private var lastTotalBytes = 0L
    private val packetSizes = mutableListOf<Long>()
    private val packetIntervals = mutableListOf<Long>()
    
    companion object {
        private const val CHANNEL_ID = "VIDEO_TRAFFIC_MONITOR"
        private const val NOTIFICATION_ID = 1
        private const val MONITORING_INTERVAL = 1000L // 1 second for more responsive updates
        private const val TAG = "TrafficMonitorService"
    }

    inner class LocalBinder : Binder() {
        fun getService(): TrafficMonitorService = this@TrafficMonitorService
    }

    override fun onCreate() {
        super.onCreate()
        try {
            android.util.Log.d(TAG, "Creating TrafficMonitorService")
            createNotificationChannel()
            classifier = VideoTrafficClassifier(this)
            startForeground(NOTIFICATION_ID, createNotification())
            startMonitoring()
            android.util.Log.d(TAG, "TrafficMonitorService created successfully")
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Service creation failed", e)
            // Try to send a simple notification instead of crashing
            try {
                createNotificationChannel()
                startForeground(NOTIFICATION_ID, createErrorNotification(e.message))
            } catch (notificationError: Exception) {
                android.util.Log.e(TAG, "Failed to create error notification", notificationError)
            }
            stopSelf()
        }
    }

    override fun onBind(intent: Intent): IBinder {
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMonitoring()
        serviceScope.cancel()
    }

    fun startMonitoring() {
        if (monitoringJob?.isActive == true) return
        
        startTime = System.currentTimeMillis()
        lastMeasurementTime = startTime
        lastTotalBytes = getCurrentTotalBytes()
        
        monitoringJob = serviceScope.launch {
            while (isActive) {
                try {
                    analyzeTraffic()
                    delay(MONITORING_INTERVAL)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    fun stopMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = null
        stopSelf()
    }
    
    fun isMonitoring(): Boolean {
        return monitoringJob?.isActive == true
    }

    private suspend fun analyzeTraffic() {
        try {
            val currentTime = System.currentTimeMillis()
            val currentTotalBytes = getCurrentTotalBytes()
            
            val timeDelta = currentTime - lastMeasurementTime
            val bytesDelta = currentTotalBytes - lastTotalBytes
            
            // Always update stats, even with no traffic
            withContext(Dispatchers.Main) {
                _trafficStats.value = AppTrafficStats(
                    bytesMonitored = totalBytesMonitored,
                    packetsAnalyzed = packetsAnalyzed,
                    averageBitrate = calculateAverageBitrate(),
                    averagePacketSize = calculateAveragePacketSize()
                )
            }
            
            if (timeDelta > 0 && bytesDelta > 0) {
                totalBytesMonitored += bytesDelta
                packetsAnalyzed++
                
                // Estimate packet size (simplified)
                val estimatedPacketSize = bytesDelta
                packetSizes.add(estimatedPacketSize)
                if (packetSizes.size > 100) packetSizes.removeAt(0) // Keep last 100
                
                // Calculate packet interval
                packetIntervals.add(timeDelta)
                if (packetIntervals.size > 100) packetIntervals.removeAt(0)
                
                // Extract features for classification
                val features = extractTrafficFeatures(bytesDelta, timeDelta)
                
                // Classify traffic
                val result = classifier.classify(features)
                
                // Update LiveData on main thread
                withContext(Dispatchers.Main) {
                    _classificationResult.value = result
                }
                
                android.util.Log.d(TAG, "Traffic analyzed: ${bytesDelta} bytes, ${result.classification}")
            } else {
                // No new traffic, send unknown result to keep UI updated
                withContext(Dispatchers.Main) {
                    _classificationResult.value = ClassificationResult(
                        ClassificationResult.Classification.UNKNOWN,
                        0.5f
                    )
                }
            }
            
            lastMeasurementTime = currentTime
            lastTotalBytes = currentTotalBytes
            
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error analyzing traffic", e)
            // Continue monitoring even if analysis fails
            withContext(Dispatchers.Main) {
                _classificationResult.value = ClassificationResult(
                    ClassificationResult.Classification.UNKNOWN,
                    0.5f
                )
            }
        }
    }

    private fun extractTrafficFeatures(bytesDelta: Long, timeDelta: Long): TrafficFeatures {
        val bitrate = if (timeDelta > 0) (bytesDelta * 8.0f * 1000) / timeDelta else 0f
        val packetSize = bytesDelta.toFloat()
        val packetInterval = timeDelta.toFloat()
        
        // Calculate statistics from recent data
        val avgPacketSize = if (packetSizes.isNotEmpty()) {
            packetSizes.average().toFloat()
        } else packetSize
        
        // Use avgPacketSize for enhanced feature calculation
        val normalizedPacketSize = if (avgPacketSize > 0) packetSize / avgPacketSize else 1f
        
        val avgPacketGap = if (packetIntervals.isNotEmpty()) {
            packetIntervals.average().toFloat()
        } else packetInterval
        
        val packetSizeVariation = if (packetSizes.size > 1) {
            val mean = packetSizes.average()
            val variance = packetSizes.map { (it - mean) * (it - mean) }.average()
            sqrt(variance).toFloat()
        } else 0f
        
        // Calculate burstiness (simplified)
        val burstiness = if (packetIntervals.size > 5) {
            val sortedIntervals = packetIntervals.sorted()
            val median = sortedIntervals[sortedIntervals.size / 2].toFloat()
            if (median > 0) packetInterval / median else 1f
        } else 1f
        
        // Simplified protocol ratios (would need deeper packet inspection in real implementation)
        val tcpRatio = 0.8f + Random.nextFloat() * 0.2f // Simulated
        val udpRatio = 1f - tcpRatio
        
        val connectionDuration = (System.currentTimeMillis() - startTime).toFloat()
        val dataVolume = totalBytesMonitored.toFloat()
        
        return TrafficFeatures(
            packetSize = packetSize,
            bitrate = bitrate,
            packetInterval = packetInterval,
            burstiness = burstiness,
            tcpRatio = tcpRatio,
            udpRatio = udpRatio,
            averagePacketGap = avgPacketGap,
            packetSizeVariation = packetSizeVariation,
            connectionDuration = connectionDuration,
            dataVolume = dataVolume
        )
    }

    private fun getCurrentTotalBytes(): Long {
        // Use system traffic stats (this monitors device-wide traffic)
        return TrafficStats.getTotalRxBytes() + TrafficStats.getTotalTxBytes()
    }

    private fun calculateAverageBitrate(): Float {
        val duration = System.currentTimeMillis() - startTime
        return if (duration > 0) {
            (totalBytesMonitored * 8.0f * 1000) / duration
        } else 0f
    }

    private fun calculateAveragePacketSize(): Float {
        return if (packetsAnalyzed > 0) {
            totalBytesMonitored.toFloat() / packetsAnalyzed
        } else 0f
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Video Traffic Monitor",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Monitors network traffic for video classification"
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Video Traffic Classifier")
            .setContentText("Monitoring network traffic...")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
    
    private fun createErrorNotification(errorMessage: String?): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Video Traffic Classifier - Error")
            .setContentText("Service error: ${errorMessage ?: "Unknown error"}")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
}
