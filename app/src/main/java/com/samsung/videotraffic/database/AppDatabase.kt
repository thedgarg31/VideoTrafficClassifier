package com.samsung.videotraffic.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import com.samsung.videotraffic.database.dao.ClassificationRecordDao
import com.samsung.videotraffic.database.dao.MonitoringSessionDao
import com.samsung.videotraffic.database.entity.ClassificationRecord
import com.samsung.videotraffic.database.entity.MonitoringSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Database(
    entities = [MonitoringSession::class, ClassificationRecord::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    
    abstract fun sessionDao(): MonitoringSessionDao
    abstract fun recordDao(): ClassificationRecordDao
    
    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null
        
        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "video_traffic_database"
                )
                .addCallback(object : RoomDatabase.Callback() {
                    override fun onCreate(db: SupportSQLiteDatabase) {
                        super.onCreate(db)
                        // Pre-populate database with any initial data if needed
                    }
                    
                    override fun onOpen(db: SupportSQLiteDatabase) {
                        super.onOpen(db)
                        // Clean up old data on database open
                        INSTANCE?.let { database ->
                            CoroutineScope(Dispatchers.IO).launch {
                                val thirtyDaysAgo = System.currentTimeMillis() - (30 * 24 * 60 * 60 * 1000L)
                                database.sessionDao().deleteOldSessions(thirtyDaysAgo)
                                database.recordDao().deleteOldRecords(thirtyDaysAgo)
                            }
                        }
                    }
                })
                .build()
                INSTANCE = instance
                instance
            }
        }
        
        fun closeDatabase() {
            INSTANCE?.close()
            INSTANCE = null
        }
    }
}
