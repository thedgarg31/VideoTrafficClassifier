package com.samsung.videotraffic.service

import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.IBinder
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.samsung.videotraffic.model.BatteryStats
import java.util.*

class BatteryMonitorService : Service() {
    private var batteryReceiver: BroadcastReceiver? = null
    private var sessionStartTime: Date? = null
    private var initialBatteryLevel: Int = 0
    private var currentBatteryLevel: Int = 0
    private var isCharging: Boolean = false
    private var temperature: Float = 0f
    private var voltage: Float = 0f
    
    private val _batteryStats = MutableLiveData<BatteryStats>()
    val batteryStats: LiveData<BatteryStats> = _batteryStats
    
    private val _batteryLevel = MutableLiveData<Int>()
    val batteryLevel: LiveData<Int> = _batteryLevel
    
    private val _isCharging = MutableLiveData<Boolean>()
    val isChargingLive: LiveData<Boolean> = _isCharging
    
    companion object {
        private const val TAG = "BatteryMonitorService"
        private const val BATTERY_UPDATE_INTERVAL = 30000L // 30 seconds
    }
    
    override fun onCreate() {
        super.onCreate()
        setupBatteryReceiver()
        getInitialBatteryInfo()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "START_MONITORING" -> startBatteryMonitoring()
            "STOP_MONITORING" -> stopBatteryMonitoring()
        }
        return START_STICKY
    }
    
    private fun setupBatteryReceiver() {
        batteryReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                when (intent?.action) {
                    Intent.ACTION_BATTERY_CHANGED -> {
                        updateBatteryInfo(intent)
                    }
                }
            }
        }
        
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_BATTERY_CHANGED)
        }
        registerReceiver(batteryReceiver, filter)
    }
    
    private fun getInitialBatteryInfo() {
        val batteryManager = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        initialBatteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        currentBatteryLevel = initialBatteryLevel
        
        val status = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS)
        isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING || 
                    status == BatteryManager.BATTERY_STATUS_FULL
        
        temperature = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_TEMPERATURE) / 10f
        voltage = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_VOLTAGE) / 1000f
        
        _batteryLevel.value = currentBatteryLevel
        _isCharging.value = isCharging
    }
    
    private fun updateBatteryInfo(intent: Intent) {
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        
        if (level != -1 && scale != -1) {
            currentBatteryLevel = (level * 100 / scale.toFloat()).toInt()
            _batteryLevel.value = currentBatteryLevel
        }
        
        val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING || 
                    status == BatteryManager.BATTERY_STATUS_FULL
        _isCharging.value = isCharging
        
        temperature = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10f
        voltage = intent.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) / 1000f
        
        updateBatteryStats()
    }
    
    private fun startBatteryMonitoring() {
        sessionStartTime = Date()
        initialBatteryLevel = currentBatteryLevel
        updateBatteryStats()
    }
    
    private fun stopBatteryMonitoring() {
        updateBatteryStats()
    }
    
    private fun updateBatteryStats() {
        val sessionId = sessionStartTime?.time?.toString() ?: return
        val startTime = sessionStartTime ?: return
        
        val batteryStats = BatteryStats(
            sessionId = sessionId,
            startTime = startTime,
            endTime = Date(),
            initialBatteryLevel = initialBatteryLevel,
            currentBatteryLevel = currentBatteryLevel,
            batteryDrainPercentage = ((initialBatteryLevel - currentBatteryLevel).toFloat() / initialBatteryLevel.toFloat()) * 100,
            monitoringDurationMinutes = (Date().time - startTime.time) / (1000 * 60),
            batteryDrainPerMinute = 0f, // Will be calculated
            isCharging = isCharging,
            temperature = temperature,
            voltage = voltage
        )
        
        _batteryStats.value = batteryStats
    }
    
    override fun onDestroy() {
        super.onDestroy()
        batteryReceiver?.let {
            unregisterReceiver(it)
        }
    }
}
