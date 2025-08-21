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
)
