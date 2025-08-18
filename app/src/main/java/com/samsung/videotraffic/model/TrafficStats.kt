package com.samsung.videotraffic.model

data class TrafficStats(
    val bytesMonitored: Long = 0,
    val packetsAnalyzed: Int = 0,
    val averageBitrate: Float = 0f,
    val averagePacketSize: Float = 0f
)
