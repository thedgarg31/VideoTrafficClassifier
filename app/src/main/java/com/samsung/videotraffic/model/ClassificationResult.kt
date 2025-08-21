package com.samsung.videotraffic.model

data class ClassificationResult(
    val classification: Classification,
    val confidence: Float,
    val timestamp: Long = System.currentTimeMillis()
) {
    enum class Classification {
        REEL,
        NON_REEL,
        UNKNOWN
    }
}
