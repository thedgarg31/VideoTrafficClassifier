package com.samsung.videotraffic.service

import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.samsung.videotraffic.model.BatteryStats
import kotlinx.coroutines.*
import java.util.*
import java.util.concurrent.TimeUnit

class BatteryMonitorService : Service() {
    private var batteryReceiver: BroadcastReceiver? = null
    private var sessionStartTime: Date? = null
    private var initialBatteryLevel: Int = 0
    private var currentBatteryLevel: Int = 0
    private var isCharging: Boolean = false
    private var temperature: Float = 0f
    private var voltage: Float = 0f
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _batteryStats = MutableLiveData<BatteryStats>()
    val batteryStats: LiveData<BatteryStats> = _batteryStats

    private val _batteryLevel = MutableLiveData<Int>()
    val batteryLevel: LiveData<Int> = _batteryLevel

    private val _isCharging = MutableLiveData<Boolean>()
    val isChargingLive: LiveData<Boolean> = _isCharging

    private var updateJob: Job? = null

    companion object {
        private const val TAG = "BatteryMonitorService"
        private const val BATTERY_UPDATE_INTERVAL = 30000L // 30 seconds
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "BatteryMonitorService created.")
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
                if (intent?.action == Intent.ACTION_BATTERY_CHANGED) {
                    updateBatteryInfo(intent)
                }
            }
        }
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        registerReceiver(batteryReceiver, filter)
    }

    private fun getInitialBatteryInfo() {
        val batteryManager = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        initialBatteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        currentBatteryLevel = initialBatteryLevel

        val status = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS)
        isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL

        _batteryLevel.postValue(currentBatteryLevel)
        _isCharging.postValue(isCharging)
    }

    private fun updateBatteryInfo(intent: Intent) {
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)

        if (level != -1 && scale != -1) {
            currentBatteryLevel = (level * 100 / scale.toFloat()).toInt()
            _batteryLevel.postValue(currentBatteryLevel)
        }

        val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
        _isCharging.postValue(isCharging)

        // Correctly read temperature and voltage from the intent extras
        temperature = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10f
        voltage = intent.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) / 1000f

        updateBatteryStats()
    }

    private fun startBatteryMonitoring() {
        if (updateJob?.isActive == true) {
            Log.d(TAG, "Battery monitoring is already running.")
            return
        }

        sessionStartTime = Date()
        initialBatteryLevel = currentBatteryLevel

        // Start a periodic coroutine to update the UI
        updateJob = serviceScope.launch {
            while (isActive) {
                updateBatteryStats()
                delay(BATTERY_UPDATE_INTERVAL)
            }
        }
        Log.d(TAG, "Started battery monitoring job.")
    }

    private fun stopBatteryMonitoring() {
        updateJob?.cancel()
        updateBatteryStats() // Final update
        Log.d(TAG, "Stopped battery monitoring job.")
    }

    private fun updateBatteryStats() {
        // Only update if a session is active
        sessionStartTime?.let { startTime ->
            val sessionId = startTime.time.toString()
            val endTime = Date()
            val durationMs = endTime.time - startTime.time
            val durationMin = TimeUnit.MILLISECONDS.toMinutes(durationMs)
            val drainPercentage = if (initialBatteryLevel > 0) {
                ((initialBatteryLevel - currentBatteryLevel).toFloat() / initialBatteryLevel.toFloat()) * 100
            } else 0f
            val drainPerMinute = if (durationMin > 0) drainPercentage / durationMin else 0f

            val batteryStats = BatteryStats(
                sessionId = sessionId,
                startTime = startTime,
                endTime = endTime,
                initialBatteryLevel = initialBatteryLevel,
                currentBatteryLevel = currentBatteryLevel,
                batteryDrainPercentage = drainPercentage,
                monitoringDurationMinutes = durationMin,
                batteryDrainPerMinute = drainPerMinute,
                isCharging = isCharging,
                temperature = temperature,
                voltage = voltage
            )
            _batteryStats.postValue(batteryStats)
            Log.d(TAG, "Updated battery stats: ${batteryStats.monitoringDurationMinutes} min, ${batteryStats.batteryDrainPercentage}% drained.")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        updateJob?.cancel()
        serviceScope.cancel()
        batteryReceiver?.let { unregisterReceiver(it) }
        Log.d(TAG, "BatteryMonitorService destroyed.")
    }
}