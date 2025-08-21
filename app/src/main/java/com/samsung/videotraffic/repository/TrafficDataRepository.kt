package com.samsung.videotraffic.repository

import android.content.Context
import android.os.Build
import android.provider.Settings
import androidx.lifecycle.LiveData
import com.samsung.videotraffic.database.AppDatabase
import com.samsung.videotraffic.database.dao.ClassificationRecordDao
import com.samsung.videotraffic.database.dao.MonitoringSessionDao
import com.samsung.videotraffic.database.entity.ClassificationRecord
import com.samsung.videotraffic.database.entity.MonitoringSession
import com.samsung.videotraffic.model.ClassificationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class TrafficDataRepository private constructor(context: Context) {
    
    private val database = AppDatabase.getDatabase(context)
    private val sessionDao: MonitoringSessionDao = database.sessionDao()
    private val recordDao: ClassificationRecordDao = database.recordDao()
    
    private val deviceId: String = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID) ?: "unknown"
    private val deviceName: String = "${Build.MANUFACTURER} ${Build.MODEL}"
    
    private var currentSessionId: String? = null
    
    companion object {
        @Volatile
        private var INSTANCE: TrafficDataRepository? = null
        
        fun getInstance(context: Context): TrafficDataRepository {
            return INSTANCE ?: synchronized(this) {
                val instance = TrafficDataRepository(context.applicationContext)
                INSTANCE = instance
                instance
            }
        }
    }
    
    // Session Management
    suspend fun startNewSession(): String {
        return withContext(Dispatchers.IO) {
            // End any existing active session
            endCurrentSession()
            
            val session = MonitoringSession(
                startTime = System.currentTimeMillis(),
                endTime = null,
                deviceId = deviceId,
                deviceName = deviceName
            )
            sessionDao.insertSession(session)
            currentSessionId = session.id
            session.id
        }
    }
    
    suspend fun endCurrentSession() {
        withContext(Dispatchers.IO) {
            currentSessionId?.let { sessionId ->
                sessionDao.endSession(sessionId, System.currentTimeMillis())
            }
            currentSessionId = null
        }
    }
    
    suspend fun getCurrentSessionId(): String? {
        return withContext(Dispatchers.IO) {
            if (currentSessionId == null) {
                val activeSession = sessionDao.getActiveSession()
                currentSessionId = activeSession?.id
            }
            currentSessionId
        }
    }
    
    suspend fun updateSessionStats(bytes: Long, packets: Int, avgBitrate: Float, peakBitrate: Float) {
        withContext(Dispatchers.IO) {
            currentSessionId?.let { sessionId ->
                sessionDao.updateSessionStats(sessionId, bytes, packets, avgBitrate, peakBitrate)
            }
        }
    }
    
    // Classification Recording
    suspend fun recordClassification(
        result: ClassificationResult,
        bytesAnalyzed: Long,
        bitrate: Float,
        packetSize: Float,
        packetInterval: Float,
        burstiness: Float,
        connectionDuration: Float,
        dataVolume: Float
    ) {
        withContext(Dispatchers.IO) {
            val sessionId = getCurrentSessionId()
            if (sessionId != null) {
                val record = ClassificationRecord.fromClassificationResult(
                    sessionId = sessionId,
                    result = result,
                    bytesAnalyzed = bytesAnalyzed,
                    bitrate = bitrate,
                    packetSize = packetSize,
                    packetInterval = packetInterval,
                    burstiness = burstiness,
                    connectionDuration = connectionDuration,
                    dataVolume = dataVolume
                )
                recordDao.insertRecord(record)
                
                // Update session counters
                when (result.classification) {
                    ClassificationResult.Classification.REEL -> sessionDao.incrementVideoDetections(sessionId)
                    ClassificationResult.Classification.NON_REEL -> sessionDao.incrementNonVideoDetections(sessionId)
                    ClassificationResult.Classification.UNKNOWN -> { /* No increment for unknown */ }
                }
            }
        }
    }
    
    // Data Retrieval
    fun getAllSessions(): LiveData<List<MonitoringSession>> = sessionDao.getAllSessions()
    
    suspend fun getAllSessionsList(): List<MonitoringSession> {
        return withContext(Dispatchers.IO) {
            sessionDao.getAllSessionsList()
        }
    }
    
    fun getSessionsByDevice(): LiveData<List<MonitoringSession>> = sessionDao.getSessionsByDevice(deviceId)
    
    fun getRecordsForSession(sessionId: String): LiveData<List<ClassificationRecord>> = 
        recordDao.getRecordsBySession(sessionId)
    
    suspend fun getSessionAnalytics(sessionId: String): List<ClassificationRecordDao.SessionAnalytics> {
        return withContext(Dispatchers.IO) {
            recordDao.getSessionAnalytics(sessionId)
        }
    }
    
    suspend fun getCurrentSession(): MonitoringSession? {
        return withContext(Dispatchers.IO) {
            currentSessionId?.let { sessionId ->
                sessionDao.getSessionById(sessionId)
            }
        }
    }
    
    // Utility methods
    fun getDeviceInfo(): Pair<String, String> = Pair(deviceId, deviceName)
    
    suspend fun cleanOldData(daysToKeep: Int = 30) {
        withContext(Dispatchers.IO) {
            val cutoffTime = System.currentTimeMillis() - (daysToKeep * 24 * 60 * 60 * 1000L)
            sessionDao.deleteOldSessions(cutoffTime)
            recordDao.deleteOldRecords(cutoffTime)
        }
    }
}
