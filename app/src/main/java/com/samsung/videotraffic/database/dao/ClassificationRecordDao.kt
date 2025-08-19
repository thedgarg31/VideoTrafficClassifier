package com.samsung.videotraffic.database.dao

import androidx.lifecycle.LiveData
import androidx.room.*
import com.samsung.videotraffic.database.entity.ClassificationRecord

@Dao
interface ClassificationRecordDao {
    
    @Query("SELECT * FROM classification_records WHERE sessionId = :sessionId ORDER BY timestamp DESC")
    fun getRecordsBySession(sessionId: String): LiveData<List<ClassificationRecord>>
    
    @Query("SELECT * FROM classification_records WHERE sessionId = :sessionId AND timestamp >= :fromTime ORDER BY timestamp DESC")
    fun getRecordsBySessionFromTime(sessionId: String, fromTime: Long): LiveData<List<ClassificationRecord>>
    
    @Query("SELECT * FROM classification_records WHERE classification = :classification AND sessionId = :sessionId ORDER BY timestamp DESC")
    fun getRecordsByClassification(sessionId: String, classification: String): LiveData<List<ClassificationRecord>>
    
    @Query("SELECT COUNT(*) FROM classification_records WHERE sessionId = :sessionId AND classification = :classification")
    suspend fun getClassificationCount(sessionId: String, classification: String): Int
    
    @Query("SELECT AVG(confidence) FROM classification_records WHERE sessionId = :sessionId AND classification = :classification")
    suspend fun getAverageConfidence(sessionId: String, classification: String): Float?
    
    @Query("SELECT AVG(bitrate) FROM classification_records WHERE sessionId = :sessionId")
    suspend fun getAverageBitrate(sessionId: String): Float?
    
    @Query("SELECT MAX(bitrate) FROM classification_records WHERE sessionId = :sessionId")
    suspend fun getMaxBitrate(sessionId: String): Float?
    
    @Query("SELECT SUM(bytesAnalyzed) FROM classification_records WHERE sessionId = :sessionId")
    suspend fun getTotalBytesAnalyzed(sessionId: String): Long?
    
    @Query("SELECT * FROM classification_records WHERE sessionId = :sessionId ORDER BY timestamp ASC LIMIT :limit")
    suspend fun getRecentRecords(sessionId: String, limit: Int): List<ClassificationRecord>
    
    @Insert
    suspend fun insertRecord(record: ClassificationRecord)
    
    @Insert
    suspend fun insertRecords(records: List<ClassificationRecord>)
    
    @Delete
    suspend fun deleteRecord(record: ClassificationRecord)
    
    @Query("DELETE FROM classification_records WHERE sessionId = :sessionId")
    suspend fun deleteRecordsBySession(sessionId: String)
    
    @Query("DELETE FROM classification_records WHERE timestamp < :olderThan")
    suspend fun deleteOldRecords(olderThan: Long)
    
    // Analytics queries
    @Query("""
        SELECT 
            classification,
            COUNT(*) as count,
            AVG(confidence) as avgConfidence,
            AVG(bitrate) as avgBitrate
        FROM classification_records 
        WHERE sessionId = :sessionId 
        GROUP BY classification
    """)
    suspend fun getSessionAnalytics(sessionId: String): List<SessionAnalytics>
    
    data class SessionAnalytics(
        val classification: String,
        val count: Int,
        val avgConfidence: Float,
        val avgBitrate: Float
    )
}
