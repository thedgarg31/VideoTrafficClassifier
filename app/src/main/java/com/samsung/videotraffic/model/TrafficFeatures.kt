package com.samsung.videotraffic.model

/**
 * Features extracted from network traffic for ML classification
 */
data class TrafficFeatures(
    // Basic traffic metrics
    val packetSize: Float,
    val bitrate: Float,
    val packetInterval: Float,
    val burstiness: Float,
    
    // Protocol information
    val tcpRatio: Float,
    val udpRatio: Float,
    
    // Timing features
    val averagePacketGap: Float,
    val packetSizeVariation: Float,
    
    // Application-level features
    val connectionDuration: Float,
    val dataVolume: Float
) {
    companion object {
        const val FEATURE_COUNT = 10
        
        fun toFloatArray(features: TrafficFeatures): FloatArray {
            return floatArrayOf(
                features.packetSize,
                features.bitrate,
                features.packetInterval,
                features.burstiness,
                features.tcpRatio,
                features.udpRatio,
                features.averagePacketGap,
                features.packetSizeVariation,
                features.connectionDuration,
                features.dataVolume
            )
        }
    }
}
