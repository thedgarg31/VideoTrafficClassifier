package com.samsung.videotraffic.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.*

@Entity(tableName = "monitoring_sessions")
data class MonitoringSession(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val startTime: Long,
    val endTime: Long?,
    val deviceId: String,
    val deviceName: String,
    val totalBytesMonitored: Long = 0,
    val totalPacketsAnalyzed: Int = 0,
    val videoDetections: Int = 0,
    val nonVideoDetections: Int = 0,
    val unknownDetections: Int = 0,
    val averageBitrate: Float = 0f,
    val peakBitrate: Float = 0f,
    val isActive: Boolean = true
)
