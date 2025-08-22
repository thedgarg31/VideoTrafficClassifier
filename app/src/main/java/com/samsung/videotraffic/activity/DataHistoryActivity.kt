package com.samsung.videotraffic.activity

import android.os.Bundle
import android.view.MenuItem
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import com.samsung.videotraffic.R
import com.samsung.videotraffic.adapter.SessionHistoryAdapter
import com.samsung.videotraffic.repository.TrafficDataRepository
import kotlinx.coroutines.*

class DataHistoryActivity : AppCompatActivity() {

    private lateinit var repository: TrafficDataRepository
    private lateinit var sessionAdapter: SessionHistoryAdapter
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // UI Elements
    private lateinit var recyclerView: RecyclerView
    private lateinit var progressBar: View
    private lateinit var noDataText: TextView
    private lateinit var statsCard: MaterialCardView
    private lateinit var totalSessionsText: TextView
    private lateinit var totalPacketsText: TextView
    private lateinit var totalDurationText: TextView
    private lateinit var reelTrafficText: TextView
    private lateinit var nonReelTrafficText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        android.util.Log.d("DataHistoryActivity", "onCreate started")
        
        try {
            setContentView(R.layout.activity_data_history_enhanced)
            android.util.Log.d("DataHistoryActivity", "Enhanced layout set successfully")

            initializeViews()
            setupActionBar()
            setupRecyclerView()
            setupRepository()
            loadHistoryData()
            android.util.Log.d("DataHistoryActivity", "onCreate completed successfully")
        } catch (e: Exception) {
            android.util.Log.e("DataHistoryActivity", "Error in onCreate", e)
            android.widget.Toast.makeText(this, "Error loading history: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun initializeViews() {
        recyclerView = findViewById(R.id.recyclerViewHistory)
        progressBar = findViewById(R.id.progressBar)
        noDataText = findViewById(R.id.textNoData)
        statsCard = findViewById(R.id.statsCard)
        totalSessionsText = findViewById(R.id.totalSessionsText)
        totalPacketsText = findViewById(R.id.totalPacketsText)
        totalDurationText = findViewById(R.id.totalDurationText)
        reelTrafficText = findViewById(R.id.reelTrafficText)
        nonReelTrafficText = findViewById(R.id.nonReelTrafficText)
    }

    private fun setupActionBar() {
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "📊 Traffic Analysis History"
    }

    private fun setupRecyclerView() {
        sessionAdapter = SessionHistoryAdapter { sessionId ->
            // TODO: Implement session detail view
            android.widget.Toast.makeText(this, "Session $sessionId details coming soon!", android.widget.Toast.LENGTH_SHORT).show()
        }
        recyclerView.apply {
            layoutManager = LinearLayoutManager(this@DataHistoryActivity)
            adapter = sessionAdapter
        }
    }

    private fun setupRepository() {
        repository = TrafficDataRepository.getInstance(this)
    }

    private fun loadHistoryData() {
        android.util.Log.d("DataHistoryActivity", "loadHistoryData started")
        scope.launch {
            try {
                progressBar.visibility = View.VISIBLE
                noDataText.visibility = View.GONE
                statsCard.visibility = View.GONE
                android.util.Log.d("DataHistoryActivity", "Loading sessions from database...")
                
                val sessions = withContext(Dispatchers.IO) {
                    repository.getAllSessionsList()
                }
                
                android.util.Log.d("DataHistoryActivity", "Loaded ${sessions.size} sessions")
                
                if (sessions.isEmpty()) {
                    noDataText.visibility = View.VISIBLE
                    recyclerView.visibility = View.GONE
                    statsCard.visibility = View.GONE
                    noDataText.text = "📭 No monitoring sessions found\n\nStart monitoring to collect traffic data!"
                    android.util.Log.d("DataHistoryActivity", "No sessions found, showing no data message")
                } else {
                    noDataText.visibility = View.GONE
                    recyclerView.visibility = View.VISIBLE
                    statsCard.visibility = View.VISIBLE
                    
                    // Update statistics
                    updateStatistics(sessions)
                    
                    // Update session list
                    sessionAdapter.submitList(sessions)
                    android.util.Log.d("DataHistoryActivity", "Sessions loaded into adapter")
                }
                
                progressBar.visibility = View.GONE
                
            } catch (e: Exception) {
                android.util.Log.e("DataHistoryActivity", "Error loading history data", e)
                progressBar.visibility = View.GONE
                noDataText.visibility = View.VISIBLE
                noDataText.text = "❌ Error loading data:\n${e.message}"
            }
        }
    }

    private fun updateStatistics(sessions: List<com.samsung.videotraffic.database.entity.MonitoringSession>) {
        val totalSessions = sessions.size
        val totalPackets = sessions.sumOf { it.totalPackets }
        val totalDuration = sessions.sumOf { it.durationMinutes }
        
        // Count traffic types
        val reelCount = sessions.count { it.reelTrafficCount > 0 }
        val nonReelCount = sessions.count { it.nonReelTrafficCount > 0 }
        
        totalSessionsText.text = "$totalSessions"
        totalPacketsText.text = "$totalPackets"
        totalDurationText.text = "${totalDuration}min"
        reelTrafficText.text = "$reelCount"
        nonReelTrafficText.text = "$nonReelCount"
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
