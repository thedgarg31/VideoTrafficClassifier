package com.samsung.videotraffic

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.google.android.material.button.MaterialButton
import com.samsung.videotraffic.activity.DataHistoryActivity
import com.samsung.videotraffic.service.AdvancedTrafficMonitorService
import com.samsung.videotraffic.service.BatteryMonitorService
import com.samsung.videotraffic.viewmodel.MainViewModel

class MainActivity : AppCompatActivity() {
    private lateinit var viewModel: MainViewModel
    private lateinit var statusText: TextView
    private lateinit var startButton: MaterialButton
    private lateinit var stopButton: MaterialButton
    private lateinit var batteryLevelText: TextView
    private lateinit var batteryDrainText: TextView
    private lateinit var monitoringDurationText: TextView
    private lateinit var batteryDrainPerMinText: TextView
    private lateinit var classificationText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var bytesSentText: TextView
    private lateinit var bytesReceivedText: TextView
    private lateinit var networkConditionText: TextView
    private lateinit var historyButton: MaterialButton
    private lateinit var shareButton: MaterialButton

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.FOREGROUND_SERVICE
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_enhanced)

        initializeViews()
        setupViewModel()
        setupClickListeners()
        checkPermissions()
    }

    private fun initializeViews() {
        statusText = findViewById(R.id.statusText)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        batteryLevelText = findViewById(R.id.batteryLevelText)
        batteryDrainText = findViewById(R.id.batteryDrainText)
        monitoringDurationText = findViewById(R.id.monitoringDurationText)
        batteryDrainPerMinText = findViewById(R.id.batteryDrainPerMinText)
        classificationText = findViewById(R.id.classificationText)
        confidenceText = findViewById(R.id.confidenceText)
        bytesSentText = findViewById(R.id.bytesSentText)
        bytesReceivedText = findViewById(R.id.bytesReceivedText)
        networkConditionText = findViewById(R.id.networkConditionText)
        historyButton = findViewById(R.id.historyButton)
        shareButton = findViewById(R.id.shareButton)
    }

    private fun setupViewModel() {
        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        // Observe traffic monitoring status
        viewModel.isMonitoring.observe(this) { isMonitoring ->
            updateMonitoringStatus(isMonitoring)
        }

        // Observe classification results
        viewModel.currentClassification.observe(this) { result ->
            updateClassificationDisplay(result)
        }

        // Observe network statistics
        viewModel.networkStats.observe(this) { stats ->
            updateNetworkStats(stats)
        }

        // Observe battery statistics
        viewModel.batteryStats.observe(this) { stats ->
            updateBatteryStats(stats)
        }
    }

    private fun setupClickListeners() {
        startButton.setOnClickListener {
            if (checkPermissions()) {
                startMonitoring()
            }
        }

        stopButton.setOnClickListener {
            stopMonitoring()
        }

        historyButton.setOnClickListener {
            val intent = Intent(this, DataHistoryActivity::class.java)
            startActivity(intent)
        }

        shareButton.setOnClickListener {
            shareData()
        }
    }

    private fun updateMonitoringStatus(isMonitoring: Boolean) {
        if (isMonitoring) {
            statusText.text = "Monitoring Active"
            statusText.setTextColor(ContextCompat.getColor(this, R.color.status_success))
            startButton.isEnabled = false
            stopButton.isEnabled = true
        } else {
            statusText.text = "Stopped"
            statusText.setTextColor(ContextCompat.getColor(this, R.color.status_error))
            startButton.isEnabled = true
            stopButton.isEnabled = false
        }
    }

    private fun updateClassificationDisplay(result: String?) {
        result?.let {
            classificationText.text = it
            when {
                it.contains("REEL", ignoreCase = true) -> {
                    classificationText.setTextColor(ContextCompat.getColor(this, R.color.reel_traffic_color))
                }
                it.contains("NON-REEL", ignoreCase = true) -> {
                    classificationText.setTextColor(ContextCompat.getColor(this, R.color.non_reel_traffic_color))
                }
                else -> {
                    classificationText.setTextColor(ContextCompat.getColor(this, R.color.unknown_traffic_color))
                }
            }
        } ?: run {
            classificationText.text = "No traffic detected"
            classificationText.setTextColor(ContextCompat.getColor(this, R.color.text_secondary_light))
        }
    }

    private fun updateNetworkStats(stats: String?) {
        stats?.let {
            // Parse and display network statistics
            // This would be implemented based on your TrafficStats model
            bytesSentText.text = "0 KB" // Placeholder
            bytesReceivedText.text = "0 KB" // Placeholder
            networkConditionText.text = "Normal" // Placeholder
        }
    }

    private fun updateBatteryStats(stats: com.samsung.videotraffic.model.BatteryStats?) {
        stats?.let {
            batteryLevelText.text = "${it.currentBatteryLevel}%"
            batteryDrainText.text = "${String.format("%.1f", it.batteryDrainPercentage)}%"
            monitoringDurationText.text = "${it.monitoringDurationMinutes} min"
            batteryDrainPerMinText.text = "${String.format("%.2f", it.batteryDrainPerMinute)}%/min"
        } ?: run {
            batteryLevelText.text = "--%"
            batteryDrainText.text = "--%"
            monitoringDurationText.text = "-- min"
            batteryDrainPerMinText.text = "--%/min"
        }
    }

    private fun startMonitoring() {
        // Start traffic monitoring service
        val trafficIntent = Intent(this, AdvancedTrafficMonitorService::class.java)
        trafficIntent.action = "START_MONITORING"
        startService(trafficIntent)

        // Start battery monitoring service
        val batteryIntent = Intent(this, BatteryMonitorService::class.java)
        batteryIntent.action = "START_MONITORING"
        startService(batteryIntent)

        viewModel.startMonitoring()
    }

    private fun stopMonitoring() {
        // Stop traffic monitoring service
        val trafficIntent = Intent(this, AdvancedTrafficMonitorService::class.java)
        trafficIntent.action = "STOP_MONITORING"
        startService(trafficIntent)

        // Stop battery monitoring service
        val batteryIntent = Intent(this, BatteryMonitorService::class.java)
        batteryIntent.action = "STOP_MONITORING"
        startService(batteryIntent)

        viewModel.stopMonitoring()
    }

    private fun shareData() {
        // Implement data sharing functionality
        // This could export data to CSV, share via email, etc.
    }

    private fun checkPermissions(): Boolean {
        val permissionsToRequest = mutableListOf<String>()
        
        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(permission)
            }
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsToRequest.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
            return false
        }
        
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                // All permissions granted, can start monitoring
            } else {
                // Some permissions denied
            }
        }
    }
}
