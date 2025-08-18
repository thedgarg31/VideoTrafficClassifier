package com.samsung.videotraffic

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.samsung.videotraffic.databinding.ActivityMainBinding
import com.samsung.videotraffic.model.ClassificationResult
import com.samsung.videotraffic.service.TrafficMonitorService
import java.text.DecimalFormat

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var monitorService: TrafficMonitorService? = null
    private var isServiceBound = false
    private var isMonitoring = false

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
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        checkPermissions()
    }

    private fun setupUI() {
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

        // Initialize statistics
        updateStatistics(0, 0)
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
        
        // Phone state permission is tricky, make it optional
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            permissions.add(Manifest.permission.READ_PHONE_STATE)
        }

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
        val permissions = arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.READ_PHONE_STATE
        )

        requestPermissionLauncher.launch(permissions)
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
            binding.btnToggle.text = getString(R.string.stop_classification)
            binding.tvStatus.text = getString(R.string.status_running)
            binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.samsung_blue))
        } else {
            binding.btnToggle.text = getString(R.string.start_classification)
            binding.tvStatus.text = getString(R.string.status_idle)
            binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            binding.tvResult.text = "---"
            binding.tvResult.setTextColor(ContextCompat.getColor(this, R.color.idle_state))
            binding.tvConfidence.visibility = android.view.View.GONE
        }
    }

    private fun observeClassificationResults() {
        monitorService?.classificationResult?.observe(this, Observer { result ->
            updateClassificationResult(result)
        })

        monitorService?.trafficStats?.observe(this, Observer { stats ->
            updateStatistics(stats.bytesMonitored, stats.packetsAnalyzed)
        })
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

        binding.tvBytesMonitored.text = getString(R.string.bytes_monitored, bytesStr)
        binding.tvPacketsAnalyzed.text = getString(R.string.packets_analyzed, packetsAnalyzed)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isServiceBound) {
            unbindService(serviceConnection)
        }
    }
}
