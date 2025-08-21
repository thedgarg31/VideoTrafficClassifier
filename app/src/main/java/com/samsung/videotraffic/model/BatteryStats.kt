package com.samsung.videotraffic.model

import java.util.*

data class BatteryStats(
    val sessionId: String,
    val startTime: Date,
    val endTime: Date? = null,
    val initialBatteryLevel: Int,
    val currentBatteryLevel: Int,
    val batteryDrainPercentage: Float,
    val monitoringDurationMinutes: Long,
    val batteryDrainPerMinute: Float,
    val isCharging: Boolean,
    val temperature: Float? = null,
    val voltage: Float? = null
) {
    fun getBatteryDrainPercentage(): Float {
        return ((initialBatteryLevel - currentBatteryLevel).toFloat() / initialBatteryLevel.toFloat()) * 100
    }
    
    fun getMonitoringDurationMinutes(): Long {
        val end = endTime ?: Date()
        return (end.time - startTime.time) / (1000 * 60)
    }
    
    fun getBatteryDrainPerMinute(): Float {
        val duration = getMonitoringDurationMinutes()
        return if (duration > 0) getBatteryDrainPercentage() / duration else 0f
    }
}
