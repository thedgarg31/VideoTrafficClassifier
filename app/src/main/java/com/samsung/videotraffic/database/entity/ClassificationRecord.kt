package com.samsung.videotraffic.database.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ForeignKey
import com.samsung.videotraffic.model.ClassificationResult
import java.util.*

@Entity(
    tableName = "classification_records",
    foreignKeys = [
        ForeignKey(
            entity = MonitoringSession::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE
        )
    ]
)
data class ClassificationRecord(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val sessionId: String,
    val timestamp: Long,
    val classification: String, // VIDEO, NON_VIDEO, UNKNOWN
    val confidence: Float,
    val bytesAnalyzed: Long,
    val bitrate: Float,
    val packetSize: Float,
    val packetInterval: Float,
    val burstiness: Float,
    val connectionDuration: Float,
    val dataVolume: Float
) {
    companion object {
        fun fromClassificationResult(
            sessionId: String,
            result: ClassificationResult,
            bytesAnalyzed: Long,
            bitrate: Float,
            packetSize: Float,
            packetInterval: Float,
            burstiness: Float,
            connectionDuration: Float,
            dataVolume: Float
        ): ClassificationRecord {
            return ClassificationRecord(
                sessionId = sessionId,
                timestamp = System.currentTimeMillis(),
                classification = result.classification.name,
                confidence = result.confidence,
                bytesAnalyzed = bytesAnalyzed,
                bitrate = bitrate,
                packetSize = packetSize,
                packetInterval = packetInterval,
                burstiness = burstiness,
                connectionDuration = connectionDuration,
                dataVolume = dataVolume
            )
        }
    }
}
