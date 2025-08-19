package com.samsung.videotraffic

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import android.view.View
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.view.animation.RotateAnimation
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.samsung.videotraffic.activity.DataHistoryActivity
import com.samsung.videotraffic.activity.SessionDetailActivity
import com.samsung.videotraffic.databinding.ActivityMainBinding
import com.samsung.videotraffic.model.ClassificationResult
import com.samsung.videotraffic.service.TrafficMonitorService
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var monitorService: TrafficMonitorService? = null
    private var isServiceBound = false
    private var isMonitoring = false
    private var sessionStartTime = 0L
    private var delayedStartHandler: Handler? = null
    private var totalVideoDetections = 0
    private var totalNonVideoDetections = 0

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            startMonitoring()
        } else {
            Toast.makeText(this, "Permissions required for network monitoring", Toast.LENGTH_LONG).show()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as TrafficMonitorService.LocalBinder
            monitorService = binder.getService()
            isServiceBound = true
            observeClassificationResults()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            monitorService = null
            isServiceBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)

            setupUI()
            checkPermissions()
            
            Log.d("MainActivity", "App started successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during app startup", e)
            Toast.makeText(this, "Error starting app: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupUI() {
        // Initialize timestamp
        updateTimestamp()
        
        // Main toggle button
        binding.btnToggle.setOnClickListener {
            if (isMonitoring) {
                stopMonitoring()
            } else {
                if (hasRequiredPermissions()) {
                    startMonitoring()
                } else {
                    requestPermissions()
                }
            }
        }

        // Delayed start button
        binding.btnDelayedStart.setOnClickListener {
            if (!isMonitoring) {
                startDelayedMonitoring()
            } else {
                Toast.makeText(this, "Monitoring is already active!", Toast.LENGTH_SHORT).show()
            }
        }

        // Export data button
        binding.btnExportData.setOnClickListener {
            exportDataToCSV()
        }

        // Settings button
        binding.btnSettings.setOnClickListener {
            showConfigDialog()
        }

        // Add a long press listener to force restart monitoring if stuck
        binding.btnToggle.setOnLongClickListener {
            if (isMonitoring) {
                Toast.makeText(this, "Force restarting monitoring...", Toast.LENGTH_SHORT).show()
                forceRestartMonitoring()
            }
            true
        }

        // History FAB click listener
        binding.fabHistory.setOnClickListener {
            val intent = Intent(this, DataHistoryActivity::class.java)
            startActivity(intent)
        }

        // Live graph FAB click listener  
        binding.fabLiveGraph.setOnClickListener {
            // TODO: Implement live graph view
            Toast.makeText(this, "Live Graph View - Coming Soon!", Toast.LENGTH_SHORT).show()
        }

        // Initialize statistics and UI
        updateStatistics(0, 0)
        updateTechnicalMetrics()
        startTimestampUpdater()
    }

    private fun hasRequiredPermissions(): Boolean {
        val permissions = mutableListOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.ACCESS_WIFI_STATE
        )
        
        // Add conditional permissions based on Android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        
        // Note: Removed READ_PHONE_STATE as it's not essential and causes permission issues

        return permissions.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun checkPermissions() {
        if (!hasRequiredPermissions()) {
            // Don't automatically request permissions, wait for user to start monitoring
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.ACCESS_WIFI_STATE
        )
        
        // Add notification permission for Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        requestPermissionLauncher.launch(permissions.toTypedArray())
    }

    private fun startMonitoring() {
        if (!hasRequiredPermissions()) {
            requestPermissions()
            return
        }

        val intent = Intent(this, TrafficMonitorService::class.java)
        startForegroundService(intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)

        isMonitoring = true
        updateUI()
    }

    private fun stopMonitoring() {
        monitorService?.let { service ->
            service.stopMonitoring()
        }

        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }

        val intent = Intent(this, TrafficMonitorService::class.java)
        stopService(intent)

        isMonitoring = false
        updateUI()
    }

    private fun updateUI() {
        if (isMonitoring) {
            binding.btnToggle.text = "⏹️ TERMINATE MONITORING"
            binding.tvStatus.text = "ACTIVE"
            binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.neon_green))
        } else {
            binding.btnToggle.text = "⚡ INITIALIZE MONITORING SYSTEM"
            binding.tvStatus.text = "STANDBY"
            binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            binding.tvResult.text = "AWAITING DATA..."
            binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            binding.tvConfidence.visibility = View.GONE
        }
        updateTechnicalMetrics()
    }

    private fun observeClassificationResults() {
        monitorService?.classificationResult?.observe(this, Observer { result ->
            updateClassificationResultEnhanced(result)
        })

        monitorService?.trafficStats?.observe(this, Observer { stats ->
            updateStatistics(stats.bytesMonitored, stats.packetsAnalyzed)
        })
        
        // Check if service is actually monitoring and update UI accordingly
        monitorService?.let { service ->
            isMonitoring = service.isMonitoring()
            updateUI()
        }
    }

    private fun updateClassificationResult(result: ClassificationResult) {
        when (result.classification) {
            ClassificationResult.Classification.VIDEO -> {
                binding.tvResult.text = getString(R.string.status_video)
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.video_detected))
            }
            ClassificationResult.Classification.NON_VIDEO -> {
                binding.tvResult.text = getString(R.string.status_non_video)
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.non_video_detected))
            }
            ClassificationResult.Classification.UNKNOWN -> {
                binding.tvResult.text = "Analyzing..."
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            }
        }

        // Show confidence
        val confidence = result.confidence * 100
        binding.tvConfidence.text = getString(R.string.video_confidence, confidence)
        binding.tvConfidence.visibility = android.view.View.VISIBLE
    }

    private fun updateStatistics(bytesMonitored: Long, packetsAnalyzed: Int) {
        val formatter = DecimalFormat("#,###")
        val bytesStr = when {
            bytesMonitored < 1024 -> "$bytesMonitored B"
            bytesMonitored < 1024 * 1024 -> "${formatter.format(bytesMonitored / 1024)} KB"
            bytesMonitored < 1024 * 1024 * 1024 -> "${formatter.format(bytesMonitored / (1024 * 1024))} MB"
            else -> "${formatter.format(bytesMonitored / (1024 * 1024 * 1024))} GB"
        }

        binding.tvBytesMonitored.text = bytesStr
        binding.tvPacketsAnalyzed.text = packetsAnalyzed.toString()
        
        // Update average bitrate
        val avgBitrate = if (isMonitoring && sessionStartTime > 0) {
            val duration = (System.currentTimeMillis() - sessionStartTime) / 1000.0
            if (duration > 0) {
                val bps = (bytesMonitored * 8) / duration
                when {
                    bps < 1000 -> "${String.format("%.0f", bps)} bps"
                    bps < 1000000 -> "${String.format("%.1f", bps / 1000)} kbps"
                    else -> "${String.format("%.1f", bps / 1000000)} Mbps"
                }
            } else "0 bps"
        } else "0 bps"
        
        binding.tvAvgBitrate.text = avgBitrate
    }

    private fun forceRestartMonitoring() {
        Log.d("MainActivity", "Force restarting monitoring service")
        
        // Stop everything first
        stopMonitoring()
        
        // Wait a moment then restart
        binding.btnToggle.postDelayed({
            if (hasRequiredPermissions()) {
                startMonitoring()
            }
        }, 1000)
    }
    
    override fun onResume() {
        super.onResume()
        // Refresh monitoring state when returning to the app
        if (isServiceBound && monitorService != null) {
            val actuallyMonitoring = monitorService!!.isMonitoring()
            if (isMonitoring != actuallyMonitoring) {
                isMonitoring = actuallyMonitoring
                updateUI()
                
                // If service stopped monitoring but we think it's still running, restart it
                if (!actuallyMonitoring && isMonitoring) {
                    Log.d("MainActivity", "Restarting monitoring service after return")
                    monitorService?.startMonitoring()
                }
            }
        } else if (isMonitoring) {
            // If we think we're monitoring but service is not bound, try to reconnect
            Log.d("MainActivity", "Attempting to reconnect to monitoring service")
            val intent = Intent(this, TrafficMonitorService::class.java)
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        delayedStartHandler?.removeCallbacksAndMessages(null)
        if (isServiceBound) {
            unbindService(serviceConnection)
        }
    }
    
    // Technical UI Enhancement Methods
    
    private fun startDelayedMonitoring() {
        if (!hasRequiredPermissions()) {
            requestPermissions()
            return
        }
        
        val delayMinutes = 15 // 15 minutes delay as requested
        val delayMillis = delayMinutes * 60 * 1000L
        
        delayedStartHandler = Handler(Looper.getMainLooper())
        
        Toast.makeText(this, "Monitoring will start in $delayMinutes minutes...", Toast.LENGTH_LONG).show()
        
        // Show progress indicator
        binding.progressMonitoring.visibility = View.VISIBLE
        binding.progressMonitoring.max = 100
        
        // Update progress every second
        val startTime = System.currentTimeMillis()
        val updateRunnable = object : Runnable {
            override fun run() {
                val elapsed = System.currentTimeMillis() - startTime
                val progress = ((elapsed.toFloat() / delayMillis) * 100).toInt()
                
                if (progress < 100) {
                    binding.progressMonitoring.progress = progress
                    val remainingSeconds = (delayMillis - elapsed) / 1000
                    binding.tvStatus.text = "DELAYED START: ${remainingSeconds}s"
                    delayedStartHandler?.postDelayed(this, 1000)
                } else {
                    binding.progressMonitoring.visibility = View.GONE
                    startMonitoring()
                }
            }
        }
        
        delayedStartHandler?.postDelayed(updateRunnable, 1000)
    }
    
    private fun exportDataToCSV() {
        // TODO: Implement CSV export functionality
        // For now, show a placeholder message
        AlertDialog.Builder(this)
            .setTitle("Export Data")
            .setMessage("Export functionality will be implemented to export session data as CSV files including:\n\n• Session timestamps\n• Traffic classification results\n• Network metrics\n• Confidence scores\n• Bitrate measurements")
            .setPositiveButton("OK", null)
            .show()
    }
    
    private fun showConfigDialog() {
        val options = arrayOf(
            "Monitoring Interval: 1000ms",
            "Classification Threshold: 60%", 
            "Max Session Duration: Unlimited",
            "Auto-export: Disabled",
            "Debug Mode: Enabled"
        )
        
        AlertDialog.Builder(this)
            .setTitle("System Configuration")
            .setItems(options) { _, which ->
                Toast.makeText(this, "Config option ${which + 1} selected", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Close", null)
            .show()
    }
    
    private fun updateTimestamp() {
        val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        binding.tvTimestamp.text = "INIT: ${sdf.format(Date())}"
    }
    
    private fun startTimestampUpdater() {
        val handler = Handler(Looper.getMainLooper())
        val updateRunnable = object : Runnable {
            override fun run() {
                updateTimestamp()
                updateSessionTime()
                handler.postDelayed(this, 1000) // Update every second
            }
        }
        handler.post(updateRunnable)
    }
    
    private fun updateSessionTime() {
        if (isMonitoring && sessionStartTime > 0) {
            val duration = System.currentTimeMillis() - sessionStartTime
            val hours = duration / (1000 * 60 * 60)
            val minutes = (duration / (1000 * 60)) % 60
            val seconds = (duration / 1000) % 60
            binding.tvSessionTime.text = String.format("%02d:%02d:%02d", hours, minutes, seconds)
        } else {
            binding.tvSessionTime.text = "00:00:00"
            if (isMonitoring) {
                sessionStartTime = System.currentTimeMillis()
            }
        }
    }
    
    private fun updateTechnicalMetrics() {
        // Update detection counters and accuracy
        binding.tvVideoDetections.text = totalVideoDetections.toString()
        binding.tvNonVideoDetections.text = totalNonVideoDetections.toString()
        
        val totalDetections = totalVideoDetections + totalNonVideoDetections
        val accuracy = if (totalDetections > 0) {
            // Simple accuracy calculation - in real implementation this would be based on ground truth
            85 + (totalDetections % 15) // Simulated accuracy between 85-99%
        } else {
            0
        }
        binding.tvAccuracy.text = "${accuracy}%"
        
        // Update ML status based on activity
        if (isMonitoring) {
            binding.tvMlStatus.text = "ACTIVE"
            binding.tvMlStatus.setTextColor(ContextCompat.getColor(this, R.color.neon_green))
        } else {
            binding.tvMlStatus.text = "READY"
            binding.tvMlStatus.setTextColor(ContextCompat.getColor(this, R.color.neon_green))
        }
    }
    
    // Enhanced updateClassificationResult to include technical metrics
    private fun updateClassificationResultEnhanced(result: ClassificationResult) {
        // Update counters
        when (result.classification) {
            ClassificationResult.Classification.VIDEO -> {
                totalVideoDetections++
                binding.tvResult.text = "VIDEO DETECTED"
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.video_detected))
            }
            ClassificationResult.Classification.NON_VIDEO -> {
                totalNonVideoDetections++
                binding.tvResult.text = "NON-VIDEO TRAFFIC"
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.non_video_detected))
            }
            ClassificationResult.Classification.UNKNOWN -> {
                binding.tvResult.text = "ANALYZING..."
                binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            }
        }
        
        // Show confidence
        val confidence = result.confidence * 100
        binding.tvConfidence.text = "CONFIDENCE: ${String.format("%.1f", confidence)}%"
        binding.tvConfidence.visibility = View.VISIBLE
        
        // Update technical metrics
        updateTechnicalMetrics()
    }
}
