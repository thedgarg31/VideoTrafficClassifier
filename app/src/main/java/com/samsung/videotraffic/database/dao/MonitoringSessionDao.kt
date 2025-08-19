package com.samsung.videotraffic.database.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.samsung.videotraffic.database.entity.MonitoringSession

@Dao
interface MonitoringSessionDao {
    
    @Query("SELECT * FROM monitoring_sessions ORDER BY startTime DESC")
    fun getAllSessions(): LiveData<List<MonitoringSession>>
    
    @Query("SELECT * FROM monitoring_sessions ORDER BY startTime DESC")
    suspend fun getAllSessionsList(): List<MonitoringSession>
    
    @Query("SELECT * FROM monitoring_sessions WHERE isActive = 1 LIMIT 1")
    suspend fun getActiveSession(): MonitoringSession?
    
    @Query("SELECT * FROM monitoring_sessions WHERE id = :sessionId")
    suspend fun getSessionById(sessionId: String): MonitoringSession?
    
    @Query("SELECT * FROM monitoring_sessions WHERE deviceId = :deviceId ORDER BY startTime DESC")
    fun getSessionsByDevice(deviceId: String): LiveData<List<MonitoringSession>>
    
    @Query("SELECT * FROM monitoring_sessions WHERE startTime >= :fromTime ORDER BY startTime DESC")
    fun getSessionsFromTime(fromTime: Long): LiveData<List<MonitoringSession>>
    
    @Insert
    suspend fun insertSession(session: MonitoringSession): Long
    
    @Update
    suspend fun updateSession(session: MonitoringSession)
    
    @Delete
    suspend fun deleteSession(session: MonitoringSession)
    
    @Query("DELETE FROM monitoring_sessions WHERE startTime < :olderThan")
    suspend fun deleteOldSessions(olderThan: Long)
    
    @Query("UPDATE monitoring_sessions SET isActive = 0, endTime = :endTime WHERE id = :sessionId")
    suspend fun endSession(sessionId: String, endTime: Long)
    
    @Query("UPDATE monitoring_sessions SET totalBytesMonitored = :bytes, totalPacketsAnalyzed = :packets, averageBitrate = :avgBitrate, peakBitrate = :peakBitrate WHERE id = :sessionId")
    suspend fun updateSessionStats(sessionId: String, bytes: Long, packets: Int, avgBitrate: Float, peakBitrate: Float)
    
    @Query("UPDATE monitoring_sessions SET videoDetections = videoDetections + 1 WHERE id = :sessionId")
    suspend fun incrementVideoDetections(sessionId: String)
    
    @Query("UPDATE monitoring_sessions SET nonVideoDetections = nonVideoDetections + 1 WHERE id = :sessionId")
    suspend fun incrementNonVideoDetections(sessionId: String)
    
    @Query("UPDATE monitoring_sessions SET unknownDetections = unknownDetections + 1 WHERE id = :sessionId")
    suspend fun incrementUnknownDetections(sessionId: String)
}
